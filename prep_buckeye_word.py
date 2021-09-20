import os
import glob
import argparse
import sys
import tqdm
import numpy as np
import torchaudio
import io
import math
import buckeye
import pickle

def spectralSize(wavLen):
    layers = [(10, 5, 3), (8, 4, 2), (4, 2, 1), (4, 2, 1), (4 ,2, 1)]
    for kernel, stride, padding in layers:
        wavLen = math.floor((wavLen + 2 * padding - 1 * (kernel - 1) - 1) / stride + 1)
    return wavLen

def non_word_fixer(word):
    if word[0] != "<":
        word = word.lower()
    non_voc_type = word[:4].upper()
    if non_voc_type == "<NOI": 
        word = "<NOISE>"
    if non_voc_type == "<VOC": 
        word = "<VOCNOISE>"
    if non_voc_type == "<ERR": 
        word = "<ERROR>"
    if non_voc_type == "<IVE": 
        # sometimes the interviewer's transcript is included - no idea what for, have no phones for that
        word = "<IVER>"
    elif non_voc_type == "<UNK": 
        # if unknown also have no phones
        word = "<UNKNOWN>"
    elif non_voc_type == "<CUT":
        # recover only the actual bit being said
        try:
            word = word.split("-")[1].split("=")[0].lower().strip(">")
        except:
            pass
    elif non_voc_type == "<EXT" or non_voc_type == '<HES': 
        try:
            word = word.split("-")[1].lower().strip(">")
        except:
            pass
    elif non_voc_type == "<LAU":
        # either pure laugh
        laughed_words = word.split("-")
        if len(laughed_words) == 1:
            word = "<LAUGH>"
        # or laughed word/sentence (?!)
        else:
            multi_laughed_words = laughed_words[1].split("_")
            if len(multi_laughed_words) == 1:
                word = multi_laughed_words[0].lower().strip(">") # single word - can recover?
            else:
                word = "<LAUGH + multiple words>" # multiple words - can't recover
    return word

def processDataset(audioFiles, outPath, splitLabel, phonesDict, dataset='timit', labels=None):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        rawPath (string): Directory of TIMIT.
        outPath (string): Directory to save TIMIT formatted for CPC pipeline
        split (string): Which of the subset of data to take. Either 'train' or 'test'.
    """
    word_labels = {} if labels is None else labels
    with open('/pio/scratch/1/i325922/data/BUCKEYE/raw/converted_aligned_words.txt', "r") as check: \
        checker = check.readlines()
    fileWriter = open(os.path.join(outPath, 'converted_aligned_words_labels_fixed_nonwords.txt'), "a")
    spkrfile = 0 if splitLabel == 'train' else 231
    for wavFile in tqdm.tqdm(audioFiles):
        if dataset == 'timit':
            labelsFile = wavFile[:-4] + '.PHN'
            with open(labelsFile) as fileReader:
                rawLabels = fileReader.readlines()
        elif dataset == 'buckeye':
            print(f"Currently working on {wavFile}")
            current_track = buckeye.Track(name=wavFile[:-4],
                              words=wavFile[:-4] + '.words',
                              phones=wavFile[:-4] + '.phones',
                              log=wavFile[:-4] + '.log',
                              txt=wavFile[:-4] + '.txt')
        waveData, samplingRate = torchaudio.load(wavFile)

        speakerName = os.path.basename(os.path.dirname(wavFile))
        trackName = os.path.basename(wavFile)
        speakerDir = os.path.join(outPath, splitLabel, speakerName)
        os.makedirs(speakerDir, exist_ok=True)
        fileWriter.write(speakerName + '-' + trackName[:-4] + ' ')
        intervals2Keep = np.arange(waveData.size(1) + 1)
        phones = []
        phoneDurations = []
        prevCode = 0
        possible_codes = np.arange(5)
        for i, l in enumerate(current_track.words):
            if dataset == 'timit':
                t0, t1, phoneCode = l.strip().split()  
            elif dataset == 'buckeye':
                try:
                    w = l.orthography   # type buckeye.Word
                except:
                    w = l.entry         # type buckeye.Pause
                w = non_word_fixer(w)
                if w[:4].upper() == "<EXC" or w is None:
                    pass
                else:
                    try:
                        temp = word_labels[w]
                    except:
                        word_labels[w] = len(word_labels.keys())
                    wordCode = str(word_labels[w])
                for p in l.phones:
                    phoneCode, t0, t1 = p.seg, p.beg, p.end
                    t0 *= samplingRate
                    t1 *= samplingRate
                    t0 = int(t0)
                    t1 = int(t1)
                    phoneDuration = t1 - t0
                    if phoneCode in ['ERROR', '<EXCLUDE-name>', '<exclude-Name>', 'EXCLUDE', '<EXCLUDE>'] or phoneCode is None:
                        intervals2Keep = intervals2Keep[~np.isin(intervals2Keep, np.arange(t0, t1 + 1))]
                    elif phoneCode in ['pau', 'epi', '1', '2', 'h#', 'IVER y'] or phoneCode.isupper():
                        nonSpeech2Keep = min(320, t1 - t0)
                        intervals2Keep = intervals2Keep[~np.isin(intervals2Keep, np.arange(t0 + nonSpeech2Keep, t1 + 1))]
                        # phones += [phoneCode] * nonSpeech2Keep
                        phones.append(wordCode)
                        phoneDurations.append(nonSpeech2Keep / samplingRate)
                    else:
                        # phones += [phoneCode] * phoneDuration
                        phones.append(wordCode)
                        phoneDurations.append(phoneDuration / samplingRate)
        initOffset = int(rawLabels[0].split()[0]) if dataset == 'timit' else int(current_track.words[0].beg * samplingRate)
        endIdx = int(rawLabels[-1].split()[-2]) if dataset == 'timit' else int(current_track.words[-1].end * samplingRate)
        intervals2Keep = intervals2Keep[intervals2Keep >= initOffset]
        intervals2Keep = intervals2Keep[intervals2Keep < min(waveData.size(1), endIdx)]
        waveData = waveData[:, intervals2Keep].view(1, -1)
        audioLen = waveData.size(1)
        spectralLen = spectralSize(audioLen)
        wordBoundaries = np.cumsum(phoneDurations)
        tDownsampled = np.linspace(0, wordBoundaries[-1], num=spectralLen) # + (audioLen / (samplingRate * spectralLen)) / 2
        downsampledLabel = []
        i = 0
        for t in tDownsampled:
            if t >= wordBoundaries[i] and i < (len(phones) - 1):
                i += 1
            downsampledLabel.append(phones[i])
        assert len(downsampledLabel) == spectralLen
        # TRAIN
        # s0502a 21130 21156, s0802b 37332 37331, s0901a 32647 32701,
        # s1003a 34821 34824, s1103a 30957 30973, s1104b 28340 28342, 
        # s1203b 33217 33219, s1803b 35929 35928, s1904a 31614 31789,
        # s2001a 16388 16389, s2203a 22794 22793, s2401a 25493 25520,
        # s2403b 30206 30214, s2603b 18019 18023, s2702b 32423 32424, 
        # s2801a 33025 33027, s2902b 32117 32119, s3202b 42691 42693,
        # s3501b 26645 26642, s3502b 22943 22942, s3601b 18547 18549, 
        # s3602b 30930 30929, s3603a 19731 19730,

        # TEST
        # s3402b 25388 25390, s4003a 24259 24258
        fileWriter.write(' '.join(downsampledLabel))
        assert len(downsampledLabel) == len(checker[spkrfile].split(" ")) - 1
        if len(downsampledLabel) == len(checker[spkrfile].split(" ")) - 1:
            print(f"All good with {trackName}!")
        else:
            print(f"Need to reprocess {trackName}")
            # torchaudio.save(os.path.join('/pio/scratch/1/i325922/data/BUCKEYE/raw/word_preprocess/' + speakerName + '-' + trackName), 
            #                     waveData, samplingRate, channels_first=True)
        spkrfile += 1
        fileWriter.write('\n')
    fileWriter.close()
    return word_labels


def getPhonesDict(phonesDocPath):
    with open(phonesDocPath) as f:
        label = f.readlines()
    phones = {}
    for l in label[42:]:
        lineWithoutSpaces = l.strip().split()
        if len(lineWithoutSpaces) > 1 and len(lineWithoutSpaces[0]) <= 4:
            phone = lineWithoutSpaces[0] 
            phones[phone] = len(phones)
            if phone in ['b', 'd', 'g', 'p', 't', 'k']:
                phones[phone + 'cl'] = len(phones)
    return phones

class Phone(object):
    def __init__(self, seg, beg=None, end=None):
        self._seg = seg
        self._beg = beg
        self._end = end

    def __repr__(self):
        return 'Phone({}, {}, {})'.format(repr(self._seg), self._beg, self._end)

    def __str__(self):
        return '<Phone [{}] at {}>'.format(self._seg, self._beg)

    @property
    def seg(self):
        """Label for this phone (e.g., using ARPABET transcription)."""
        return self._seg

    @property
    def beg(self):
        """Timestamp where the phone begins."""
        return self._beg

    @property
    def end(self):
        """Timestamp where the phone ends."""
        return self._end

    @property
    def dur(self):
        """Duration of the phone."""
        try:
            return self._end - self._beg

        except TypeError:
            raise AttributeError('Duration is not available if beg and end '
                                 'are not numeric types')

def process_phones(phones):
    # skip the header
    line = phones.readline()

    while not line.startswith('#'):
        if line == '':
            raise EOFError

        line = phones.readline()

    line = phones.readline()

    # iterate over entries
    previous = 0.0
    while line != '':
        try:
            time, color, phone = line.split(None, 2)

            if '+1' in phone:
                phone = phone.replace('+1', '')

            if ';' in phone:
                phone = phone.split(';')[0]

            phone = phone.strip()

        except ValueError:
            if line == '\n':
                line = phones.readline()
                continue

            time, color = line.split()
            phone = None

        time = float(time)
        yield Phone(phone, previous, time)

        previous = time
        line = phones.readline()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Prepare the TIMIT data set for the CPC pipeline.')
    parser.add_argument('--pathDB', type=str, default="/pio/scratch/1/i325922/data/BUCKEYE/raw",
                        help='Path to the directory containing the audio '
                        'files')
    parser.add_argument("--pathOut", type=str, default="/pio/scratch/1/i325922/data/BUCKEYE/raw",
                        help='Path out the output directory')
    parser.add_argument("--dataset", type=str, default='buckeye',
                        help='The dataset to be processed, namely timit or buckeye')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.pathOut, exist_ok=True)
    ext = ".WAV" if args.dataset =='timit' else ".wav"
    audioFilesTrain = glob.glob(os.path.join(args.pathDB, "TRAIN/**/*" + ext), recursive=True)
    open(os.path.join(args.pathOut, 'converted_aligned_words_labels_fixed_nonwords.txt'), "w").close()
    print("Creating train set")
    labels = processDataset(audioFilesTrain, args.pathOut, 'train', None, args.dataset)
    print("Train set ready!")
    
    audioFilesTest = glob.glob(os.path.join(args.pathDB, "TEST/**/*" + ext), recursive=True)
    print("Creating test set")
    labels = processDataset(audioFilesTest, args.pathOut, 'test', None, args.dataset, labels=labels)
    with open(args.pathOut + '/word_labels.pickle', 'wb') as handle:
            pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Test set ready!")
    print ("Data preparation is complete !")

if __name__ == '__main__':
    main(sys.argv[1:])