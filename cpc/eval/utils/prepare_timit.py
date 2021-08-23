import os
import glob
import argparse
import sys
import tqdm
import numpy as np
import torchaudio
import io
import math

def spectralSize(wavLen):
    layers = [(10, 5, 3), (8, 4, 2), (4, 2, 1), (4, 2, 1), (4 ,2, 1)]
    for kernel, stride, padding in layers:
        wavLen = math.floor((wavLen + 2 * padding - 1 * (kernel - 1) - 1) / stride + 1)
    return wavLen


def processDataset(audioFiles, outPath, splitLabel, phonesDict, dataset='timit'):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        rawPath (string): Directory of TIMIT.
        outPath (string): Directory to save TIMIT formatted for CPC pipeline
        split (string): Which of the subset of data to take. Either 'train' or 'test'.
    """
    # Remove all 'SA' records.
    # audioFiles = [p for p in audioFiles if 'SA' not in os.path.basename(p)]
    fileWriter = open(os.path.join(outPath, 'converted_aligned_phones.txt'), "a")
    for wavFile in tqdm.tqdm(audioFiles):
        if dataset == 'timit':
            labelsFile = wavFile[:-4] + '.PHN'
            with open(labelsFile) as fileReader:
                rawLabels = fileReader.readlines()
        elif dataset == 'buckeye':
            print(f"Currently working on {wavFile}")
            labelsFile = wavFile[:-4] + '.phones'
            if not hasattr(labelsFile, 'readline'):
                phones = io.open(labelsFile, encoding='latin-1')
            rawLabels = list(process_phones(phones))
            phones.close()
        waveData, samplingRate = torchaudio.load(wavFile)

        speakerName = os.path.basename(os.path.dirname(wavFile))
        trackName = os.path.basename(wavFile)
        speakerDir = os.path.join(outPath, splitLabel, speakerName)
        os.makedirs(speakerDir, exist_ok=True)
        
        fileWriter.write(speakerName + '-' + trackName[:-4] + ' ')
        intervals2Keep = np.arange(waveData.size(1) + 1)
        phones = []
        phoneDurations = []
        for i, l in enumerate(rawLabels):
            if dataset == 'timit':
                t0, t1, phoneCode = l.strip().split()  
            elif dataset == 'buckeye':
                t0, t1, phoneCode = l.beg, l.end, l.seg
                t0 *= samplingRate
                t1 *= samplingRate
            t0 = int(t0)
            t1 = int(t1)
            phoneDuration = t1 - t0
            if phoneCode in ['ERROR', '<EXCLUDE-name>', '<exclude-Name>', 'EXCLUDE', '<EXCLUDE>'] or phoneCode is None:
                intervals2Keep = intervals2Keep[~np.isin(intervals2Keep, np.arange(t0, t1 + 1))]
            elif phoneCode in ['pau', 'epi', '1', '2', 'h#', 'IVER y'] or phoneCode.isupper():
                phoneCode = str(phonesDict[phoneCode])
                nonSpeech2Keep = min(320, t1 - t0)
                intervals2Keep = intervals2Keep[~np.isin(intervals2Keep, np.arange(t0 + nonSpeech2Keep, t1 + 1))]
                # phones += [phoneCode] * nonSpeech2Keep
                phones.append(phoneCode)
                phoneDurations.append(nonSpeech2Keep / samplingRate)
            else:
                phoneCode = str(phonesDict[phoneCode])
                # phones += [phoneCode] * phoneDuration
                phones.append(phoneCode)
                phoneDurations.append(phoneDuration / samplingRate)
        initOffset = int(rawLabels[0].split()[0]) if dataset == 'timit' else int(rawLabels[0].beg * samplingRate)
        endIdx = int(rawLabels[-1].split()[-2]) if dataset == 'timit' else int(rawLabels[-1].end * samplingRate)
        intervals2Keep = intervals2Keep[intervals2Keep >= initOffset]
        intervals2Keep = intervals2Keep[intervals2Keep < min(waveData.size(1), endIdx)]
        waveData = waveData[:, intervals2Keep].view(1, -1)
        audioLen = waveData.size(1)
        spectralLen = spectralSize(audioLen)
        
        phoneBoundaries = np.cumsum(phoneDurations)
        tDownsampled = np.linspace(0, phoneBoundaries[-1], num=spectralLen) # + (audioLen / (samplingRate * spectralLen)) / 2
        downsampledLabel = []
        i = 0
        for t in tDownsampled:
            if t >= phoneBoundaries[i] and i < (len(phones) - 1):
                i += 1
            downsampledLabel.append(phones[i])
        assert len(downsampledLabel) == spectralLen
        fileWriter.write(' '.join(downsampledLabel))
        torchaudio.save(os.path.join(speakerDir, speakerName + '-' + trackName), waveData, samplingRate, channels_first=True)
        fileWriter.write('\n')
    fileWriter.close()


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
    parser.add_argument('--pathDB', type=str,
                        help='Path to the directory containing the audio '
                        'files')
    parser.add_argument("--pathOut", type=str,
                        help='Path out the output directory')
    parser.add_argument("--dataset", type=str,
                        help='The dataset to be processed, namely timit or buckeye')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.pathOut, exist_ok=True)
    if args.dataset == 'timit':
        phonesDict = getPhonesDict(os.path.join(args.pathDB, 'DOC', 'PHONCODE.DOC'))
    elif args.dataset == 'buckeye':
        # sadly no lookup file available
        phonesDict = {  # phonemes which should be the same for TIMIT
                        'a': 0, 'aa': 1, 'aan': 2, 'ae': 3, 'aen': 4, 'ah': 0, 'ahn': 5, 'an': 5, 
                        'ao': 6, 'aon': 7, 'aw': 8, 'awn': 9, 'ay': 10, 'ayn': 11, 'b': 12, 'ch': 13, 
                        'd': 14, 'dh': 15, 'dx': 16, 'eh': 17, 'ehn': 18, 'el': 19, 'em': 20, 'en': 21, 
                        'eng': 22, 'er': 23, 'ern': 24, 'ey': 25, 'eyn': 26, 'f': 27, 'g': 28, 'h': 29, 
                        'hh': 29, 'hhn': 30, 'i': 31, 'id': 31, 'ih': 31, 'ihn': 32, 'iy': 33, 'iyih': 32, 
                        'iyn': 32, 'jh': 34, 'k': 35, 'l': 36, 'm': 37, 'n': 38, 'ng': 39, 'nx': 40, 
                        'ow': 41, 'own': 42, 'oy': 43, 'oyn': 44, 'p': 45, 'q': 46, 'r': 47, 's': 48, 
                        'sh': 49, 't': 50, 'th': 51, 'tq': 46, 'uh': 52, 'uhn': 53, 'uw': 54, 'uwix': 54, 
                        'uwn': 55, 'v': 56, 'w': 57, 'y': 58, 'z': 59, 'zh': 60, 
                        # non-phoneme labels, treat them as either silence or noise
                        'SIL': 61, f"{{B_TRANS}}": 61, f"{{E_TRANS}}": 61, 'B_THIRD_SPKR': 61, 'E_THIRD_SPKR': 61, 
                        'NOISE': 62, 'VOCNOISE': 62, 'IVER': 62, 'LAUGH': 62, 'UNKNOWN': 62, 'CUTOFF': 62,
                        # leftovers added after looping through the whole dataset, try to match them with existing ones:
                        'x': 63, 'e': 17, 'ih l': 31, 'ah r': 0, 'ah l': 0, 'ah ix': 0, 'uw ix':54, 'ah n': 5, 
                        'no': 64, 'IVER y': 62, 'j': 34, 'IVER-LAUGH': 62
                    }
    ext = ".WAV" if args.dataset =='timit' else ".wav"
    audioFilesTrain = glob.glob(os.path.join(args.pathDB, "TRAIN/**/*" + ext), recursive=True)
    open(os.path.join(args.pathOut, 'converted_aligned_phones.txt'), "w").close()
    print("Creating train set")
    processDataset(audioFilesTrain, args.pathOut, 'train', phonesDict, args.dataset)
    print("Train set ready!")
    
    audioFilesTest = glob.glob(os.path.join(args.pathDB, "TEST/**/*" + ext), recursive=True)
    print("Creating test set")
    processDataset(audioFilesTest, args.pathOut, 'test', phonesDict, args.dataset)
    print("Test set ready!")
    print ("Data preparation is complete !")

if __name__ == '__main__':
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main(sys.argv[1:])