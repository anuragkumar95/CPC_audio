# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import random
import time
import tqdm
import torch
import subprocess
import soundfile as sf
from pathlib import Path
from copy import deepcopy
from torch.multiprocessing import Pool
from multiprocessing import dummy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler, BatchSampler
import zipfile
import torchaudio
#import librosa


class AudioBatchData(Dataset):

    def __init__(self,
                 path,
                 sizeWindow,
                 seqNames,
                 phoneLabelsDict,
                 nSpeakers,
                 nProcessLoader=50,
                 MAX_SIZE_LOADED=4000000000):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabelsDict (dictionnary): if not None, a dictionnary with the
                                             following entries

                                             "step": size of a labelled window
                                             "$SEQ_NAME": list of phonem labels for
                                             the sequence $SEQ_NAME
           - nSpeakers (int): number of speakers to expect.
           - nProcessLoader (int): number of processes to call when loading the
                                   data from the disk
           - MAX_SIZE_LOADED (int): target maximal size of the floating array
                                    containing all loaded data.
        """
        self.MAX_SIZE_LOADED = MAX_SIZE_LOADED
        self.nProcessLoader = nProcessLoader
        self.dbPath = Path(path)
        self.sizeWindow = sizeWindow
        self.seqNames = [(s, self.dbPath / x) for s, x in seqNames]
        #self.reload_pool = Pool(nProcessLoader)
        self.reload_pool = dummy.Pool(nProcessLoader)

        self.prepare()
        self.speakers = list(range(nSpeakers))
        self.data = []

        self.phoneSize = 0 if phoneLabelsDict is None else \
            phoneLabelsDict["step"]
        self.phoneStep = 0 if phoneLabelsDict is None else \
            self.sizeWindow // self.phoneSize

        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.loadNextPack(first=True)
        self.loadNextPack()
        self.doubleLabels = False

    def resetPhoneLabels(self, newPhoneLabels, step):
        self.phoneSize = step
        self.phoneStep = self.sizeWindow // self.phoneSize
        self.phoneLabelsDict = deepcopy(newPhoneLabels)
        self.loadNextPack()

    def splitSeqTags(seqName):
        path = os.path.normpath(seqName)
        return path.split(os.sep)

    def getSeqNames(self):
        return [str(x[1]) for x in self.seqNames]

    def clear(self):
        if 'data' in self.__dict__:
            del self.data
        if 'speakerLabel' in self.__dict__:
            del self.speakerLabel
        if 'phoneLabels' in self.__dict__:
            del self.phoneLabels
        if 'seqLabel' in self.__dict__:
            del self.seqLabel

    def prepare(self):
        randomstate = random.getstate()
        random.seed(767543)  # set seed only for batching so that it is random but always same for same dataset
                             # so that capturing captures data for same audio across runs if same dataset provided
        random.shuffle(self.seqNames)
        random.setstate(randomstate)  # restore random state so that other stuff changes with seed in args
        start_time = time.time()

        print("Checking length...")
        allLength = self.reload_pool.map(extractLength, self.seqNames)

        self.packageIndex, self.totSize = [], 0
        start, packageSize = 0, 0
        for index, length in tqdm.tqdm(enumerate(allLength)):
            packageSize += length
            if packageSize > self.MAX_SIZE_LOADED:
                self.packageIndex.append([start, index])
                self.totSize += packageSize
                start, packageSize = index, 0

        if packageSize > 0:
            self.packageIndex.append([start, len(self.seqNames)])
            self.totSize += packageSize

        print(f"Done, elapsed: {time.time() - start_time:.3f} seconds")
        print(f'Scanned {len(self.seqNames)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f"{len(self.packageIndex)} chunks computed")
        self.currentPack = -1
        self.nextPack = 0

    def getNPacks(self):
        return len(self.packageIndex)

    def loadNextPack(self, first=False):
        self.clear()
        if not first:
            self.currentPack = self.nextPack
            start_time = time.time()
            print('Joining pool')
            self.r.wait()
            print(f'Joined process, elapsed={time.time()-start_time:.3f} secs')
            self.nextData = self.r.get()
            self.parseNextDataBlock()
            del self.nextData
        self.nextPack = (self.currentPack + 1) % len(self.packageIndex)
        seqStart, seqEnd = self.packageIndex[self.nextPack]
        if self.nextPack == 0 and len(self.packageIndex) > 1:
            self.prepare()
        self.r = self.reload_pool.map_async(loadFile,
                                            self.seqNames[seqStart:seqEnd])

    def parseNextDataBlock(self):

        # Labels
        self.speakerLabel = [0]
        self.seqLabel = [0]
        self.phoneLabels = []
        speakerSize = 0
        indexSpeaker = 0

        # To accelerate the process a bit
        self.nextData.sort(key=lambda x: (x[0], x[1]))
        tmpData = []

        for speaker, seqName, seq in self.nextData:

            # sometimes some data may be missing
            if self.phoneLabelsDict is not None and seqName not in self.phoneLabelsDict:
                continue
            
            while self.speakers[indexSpeaker] < speaker:
                indexSpeaker += 1
                self.speakerLabel.append(speakerSize)
            if self.speakers[indexSpeaker] != speaker:
                raise ValueError(f'{speaker} invalid speaker')

            if self.phoneLabelsDict is not None:
                self.phoneLabels += self.phoneLabelsDict[seqName]
                newSize = len(self.phoneLabelsDict[seqName]) * self.phoneSize
                seq = seq[:newSize]

            sizeSeq = seq.size(0)
            tmpData.append(seq)
            self.seqLabel.append(self.seqLabel[-1] + sizeSeq)
            speakerSize += sizeSeq
            del seq

        self.speakerLabel.append(speakerSize)
        self.data = torch.cat(tmpData, dim=0)

    def getPhonem(self, idx):
        idPhone = idx // self.phoneSize
        return self.phoneLabels[idPhone:(idPhone + self.phoneStep)]

    def getSpeakerLabel(self, idx):
        idSpeaker = next(x[0] for x in enumerate(
            self.speakerLabel) if x[1] > idx) - 1
        return idSpeaker

    def __len__(self):
        # all audio is glued together, totSize is num of frames and sizeWindow is perhaps sample size
        return self.totSize // self.sizeWindow

    def __getitem__(self, idx):

        if idx < 0 or idx >= len(self.data) - self.sizeWindow - 1:
            print(idx)

        outData = self.data[idx:(self.sizeWindow + idx)].view(1, -1)
        labelData = {}
        labelData['speaker'] = torch.tensor(self.getSpeakerLabel(idx), dtype=torch.long)
        if self.phoneSize > 0:
            label_phone = torch.tensor(self.getPhonem(idx), dtype=torch.long)
            labelData['phone'] = label_phone
            # if not self.doubleLabels:
            #     label = label_phone
        # else:
        #     label_phone = torch.zeros(1)

        # if self.doubleLabels:
        #     return outData, label, label_phone

        return outData, labelData

    def getNSpeakers(self):
        return len(self.speakers)

    def getNSeqs(self):
        return len(self.seqLabel) - 1

    def getNLoadsPerEpoch(self):
        return len(self.packageIndex)

    def getBaseSampler(self, type, batchSize, offset):
        if type == "samespeaker":
            return SameSpeakerSampler(batchSize, self.speakerLabel,
                                      self.sizeWindow, offset)
        if type == "samesequence":
            return SameSpeakerSampler(batchSize, self.seqLabel,
                                      self.sizeWindow, offset)
        if type == "sequential":
            return SequentialSampler(len(self.data), self.sizeWindow,
                                     offset, batchSize)
        sampler = UniformAudioSampler(len(self.data), self.sizeWindow,
                                      offset)
        return BatchSampler(sampler, batchSize, True)

    def getDataLoader(self, batchSize, type, randomOffset, numWorkers=0,
                      onLoop=-1):
        r"""
        Get a batch sampler for the current dataset.
        Args:
            - batchSize (int): batch size
            - groupSize (int): in the case of type in ["speaker", "sequence"]
            number of items sharing a same label in the group
            (see AudioBatchSampler)
            - type (string):
                type == "speaker": grouped sampler speaker-wise
                type == "sequence": grouped sampler sequence-wise
                type == "sequential": sequential sampling
                else: uniform random sampling of the full audio
                vector
            - randomOffset (bool): if True add a random offset to the sampler
                                   at the begining of each iteration
        """
        nLoops = len(self.packageIndex)
        totSize = self.totSize // (self.sizeWindow * batchSize)
        if onLoop >= 0:
            self.currentPack = onLoop - 1
            self.loadNextPack()
            nLoops = 1

        def samplerCall():
            offset = random.randint(0, self.sizeWindow // 2) \
                if randomOffset else 0
            return self.getBaseSampler(type, batchSize, offset)

        return AudioLoader(self, samplerCall, nLoops, self.loadNextPack,
                           totSize, numWorkers)

def unzip_file(filepath):
    """
    Unzips file and stores it in the same directory
    """
    with zipfile.ZipFile(filepath, 'r') as fp:
        fp.extractall(filepath[:-4]+'.mp4')


def loadFile(data):
    speaker, fullPath = data
    seqName = fullPath.stem
    #print(speaker, seqName, fullPath)
    # Due to some issues happening when combining torchaudio.load
    # with torch.multiprocessing we use soundfile to load the data
    ### Write code here for unzipping files and read them after #####
    #unzip_file(fullPath)
    ### check what is fullpath and actual file name ###
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seq = torch.tensor(sf.read(str(fullPath))[0]).float()
        #seq = torch.tensor(librosa.load(str(fullPath))[0]).float()
        if len(seq.size()) == 2:
            seq = seq.mean(dim=1)
        return speaker, seqName, seq


class AudioLoader(object):
    r"""
    A DataLoader meant to handle an AudioBatchData object.
    In order to handle big datasets AudioBatchData works with big chunks of
    audio it loads sequentially in memory: once all batches have been sampled
    on a chunk, the AudioBatchData loads the next one.
    """
    def __init__(self,
                 dataset,
                 samplerCall,
                 nLoop,
                 updateCall,
                 size,
                 numWorkers):
        r"""
        Args:
            - dataset (AudioBatchData): target dataset
            - samplerCall (function): batch-sampler to call
            - nLoop (int): number of chunks to load
            - updateCall (function): function loading the next chunk
            - size (int): total number of batches
            - numWorkers (int): see torch.utils.data.DataLoader
        """
        self.samplerCall = samplerCall
        self.updateCall = updateCall
        self.nLoop = nLoop
        self.size = size
        self.dataset = dataset
        self.numWorkers = numWorkers

    def __len__(self):
        return self.size

    def __iter__(self):

        for i in range(self.nLoop):
            sampler = self.samplerCall()
            dataloader = DataLoader(self.dataset,
                                    batch_sampler=sampler,
                                    num_workers=self.numWorkers)
            for x in dataloader:
                yield x
            if i < self.nLoop - 1:
                self.updateCall()


class UniformAudioSampler(Sampler):

    def __init__(self,
                 dataSize,
                 sizeWindow,
                 offset):

        self.len = dataSize // sizeWindow
        self.sizeWindow = sizeWindow
        self.offset = offset
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        return iter((self.offset
                     + self.sizeWindow * torch.randperm(self.len)).tolist())

    def __len__(self):
        return self.len


class SequentialSampler(Sampler):

    def __init__(self, dataSize, sizeWindow, offset, batchSize):

        self.len = (dataSize // sizeWindow) // batchSize
        self.sizeWindow = sizeWindow
        self.offset = offset
        self.startBatches = [x * (dataSize // batchSize)
                             for x in range(batchSize)]
        self.batchSize = batchSize
        if self.offset > 0:
            self.len -= 1

    def __iter__(self):
        for idx in range(self.len):
            yield [self.offset + self.sizeWindow * idx
                   + start for start in self.startBatches]

    def __len__(self):
        return self.len


class SameSpeakerSampler(Sampler):

    def __init__(self,
                 batchSize,
                 samplingIntervals,
                 sizeWindow,
                 offset):

        self.samplingIntervals = samplingIntervals
        self.sizeWindow = sizeWindow
        self.batchSize = batchSize
        self.offset = offset

        if self.samplingIntervals[0] != 0:
            raise AttributeError("Sampling intervals should start at zero")

        nWindows = len(self.samplingIntervals) - 1
        self.sizeSamplers = [(self.samplingIntervals[i+1] -
                              self.samplingIntervals[i]) // self.sizeWindow
                             for i in range(nWindows)]

        if self.offset > 0:
            self.sizeSamplers = [max(0, x - 1) for x in self.sizeSamplers]

        order = [(x, torch.randperm(val).tolist())
                 for x, val in enumerate(self.sizeSamplers) if val > 0]

        # Build Batches
        self.batches = []
        for indexSampler, randperm in order:
            indexStart, sizeSampler = 0, self.sizeSamplers[indexSampler]
            while indexStart < sizeSampler:
                indexEnd = min(sizeSampler, indexStart + self.batchSize)
                locBatch = [self.getIndex(x, indexSampler)
                            for x in randperm[indexStart:indexEnd]]
                indexStart = indexEnd
                self.batches.append(locBatch)

    def __len__(self):
        return len(self.batches)

    def getIndex(self, x, iInterval):
        return self.offset + x * self.sizeWindow \
            + self.samplingIntervals[iInterval]

    def __iter__(self):
        random.shuffle(self.batches)
        return iter(self.batches)

def convert_to_wav(fin, fout):
    """
    Reads the file from fin and saves the file in wav format in fout
    """
    print("Running")
    print("FIN:", fin, "FOUT:", fout)
    temp = subprocess.run(["ffmpeg",
                           "-i", 
                           fin, 
                           fout], 
                           stdout=subprocess.PIPE, 
                           stderr=subprocess.PIPE)
    print(temp.stderr)
    print(temp.stdout)
    

def extractLength(couple):
    speaker, locPath = couple
    #unzip_file(str(locPath))
    #convert_to_wav(fin=str(locPath)[:-4]+'.mp4', 
    #               fout=str(locPath)[:-4]+'.wav')
    #info = torchaudio.info(str(locPath))[0]
    torchaudio.set_audio_backend('soundfile')
    info = torchaudio.info(str(locPath))
    return info.num_frames


def findAllSeqs(dirName,
                extension='.flac',
                loadCache=False,
                speaker_level=1):
    r"""
    Lists all the sequences with the given extension in the dirName directory.
    Output:
        outSequences, speakers

        outSequence
        A list of tuples seq_path, speaker where:
            - seq_path is the relative path of each sequence relative to the
            parent directory
            - speaker is the corresponding speaker index

        outSpeakers
        The speaker labels (in order)

    The speaker labels are organized the following way
    \dirName
        \speaker_label
            \..
                ...
                seqName.extension

    Adjust the value of speaker_level if you want to choose which level of
    directory defines the speaker label. Ex if speaker_level == 2 then the
    dataset should be organized in the following fashion
    \dirName
        \crappy_label
            \speaker_label
                \..
                    ...
                    seqName.extension
    Set speaker_label == 0 if no speaker label will be retrieved no matter the
    organization of the dataset.

    """
    cache_path = os.path.join(dirName, '_seqs_cache.txt')
    if loadCache:
        try:
            outSequences, speakers = torch.load(cache_path)
            print(f'Loaded from cache {cache_path} successfully')
            return outSequences, speakers
        except OSError as err:
            print(f'Ran in an error while loading {cache_path}: {err}')
        print('Could not load cache, rebuilding')

    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    #for root, dirs, filenames in tqdm.tqdm(os.walk(dirName)):
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]

        if len(filtered_files) > 0:
            speakerStr = (os.sep).join(
                root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root[prefixSize:], filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]
    for key, index in speakersTarget.items():
        outSpeakers[index] = key
    try:
        torch.save((outSequences, outSpeakers), cache_path)
        print(f'Saved cache file at {cache_path}')
    except OSError as err:
        print(f'Ran in an error while saving {cache_path}: {err}')
    return outSequences, outSpeakers


def parseSeqLabels(pathLabels):
    with open(pathLabels, 'r') as f:
        lines = f.readlines()
    output = {"step": 160}  # Step in librispeech dataset is 160bits
    maxPhone = 0
    for line in lines:
        data = line.split()
        output[data[0]] = [int(x) for x in data[1:]]
        maxPhone = max(maxPhone, max(output[data[0]]))
    return output, maxPhone + 1


def filterSeqs(pathTxt, seqCouples, percentage=None, totalNum=None):
    assert(percentage is None or totalNum is None)
    with open(pathTxt, 'r') as f:
        inSeqs = [p.replace('\n', '') for p in f.readlines()]

    inSeqs.sort()
    seqCouples.sort(key=lambda x: os.path.basename(os.path.splitext(x[1])[0]))
    output, index = [], 0
    for x in seqCouples:
        seq = os.path.basename(os.path.splitext(x[1])[0])
        while index < len(inSeqs) and seq > inSeqs[index]:
            index += 1
        if index == len(inSeqs):
            break
        if seq == inSeqs[index]:
            output.append(x)
    if percentage is not None:
        assert(percentage < 100)
        originalOutput = output
        output = []
        for i, elem in enumerate(originalOutput):
            if (100. * len(output) / float(i+1)) < float(percentage):
                output.append(elem)
    elif totalNum is not None:
        lastCaptured = -1.
        lastIdCaptured = -1
        captureEach = max(float(len(output)) / float(totalNum), 1.)
        originalOutput = output
        output = []
        for i, elem in enumerate(originalOutput):
            toCapture = int(round(lastCaptured + captureEach))
            if i == lastIdCaptured or i < toCapture:
                continue
            lastIdCaptured = i
            lastCaptured += captureEach
            output.append(elem)
    return output
