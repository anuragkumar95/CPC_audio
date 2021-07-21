import os
import glob
import argparse
import sys
from shutil import copyfile
import tqdm
import numpy as np
import torchaudio


def processDataset(audioFiles, outPath, split, phonesDict):
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
        waveData, samplingRate = torchaudio.load(wavFile)
        speakerName = os.path.basename(os.path.dirname(wavFile))
        trackName = os.path.basename(wavFile)
        speakerDir = os.path.join(outPath, split, speakerName)
        os.makedirs(speakerDir, exist_ok=True)
        labelsFile = wavFile[:-4] + '.PHN'
        with open(labelsFile) as fileReader:
            rawTimitLabel = fileReader.readlines()
        fileWriter.write(speakerName + '-' + trackName[:-4])
        intervals2Keep = np.arange(waveData.size(1))
        for l in rawTimitLabel:
            t0, t1, phoneCode = l.strip().split()
            t0 = int(t0)
            t1 = int(t1)
            phoneDuration = t1 - t0
            if phoneCode in ['pau', 'epi', '1', '2', 'h#']:
                phoneCode = str(phonesDict[phoneCode])
                nonSpeech2Keep = min(320, t1 - t0)
                intervals2Keep = intervals2Keep[~np.isin(intervals2Keep, np.arange(t0, t0 + nonSpeech2Keep + 1))]
                subSampledNonSpeech2Keep = int(nonSpeech2Keep * 100 / samplingRate)
                fileWriter.write((' ' + phoneCode) * subSampledNonSpeech2Keep)
            else:
                phoneCode = str(phonesDict[phoneCode])
                subSampledPhoneDuration = int(phoneDuration * 100 / samplingRate)
                fileWriter.write((' ' + phoneCode) * subSampledPhoneDuration)
        waveData = waveData[:, intervals2Keep].view(1, -1)
        torchaudio.save(os.path.join(speakerDir, trackName), waveData, samplingRate, channels_first=True)
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


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Prepare the TIMIT data set for the CPC pipeline.')
    parser.add_argument('--pathDB', type=str,
                        help='Path to the directory containing the audio '
                        'files')
    parser.add_argument("--pathOut", type=str,
                        help='Path out the output directory')
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    os.makedirs(args.pathOut, exist_ok=True)
    phonesDict = getPhonesDict(os.path.join(args.pathDB, 'DOC', 'PHONCODE.DOC'))
    
    audioFilesTrain = glob.glob(os.path.join(args.pathDB, "TRAIN/**/*.WAV"), recursive=True)
    open(os.path.join(args.pathOut, 'converted_aligned_phones.txt'), "w").close()
    print("Creating train set")
    processDataset(audioFilesTrain, args.pathOut, 'train', phonesDict)
    print("Train set ready!")
    
    audioFilesTest = glob.glob(os.path.join(args.pathDB, "TEST/**/*.WAV"), recursive=True)
    print("Creating test set")
    processDataset(audioFilesTest, args.pathOut, 'test', phonesDict)
    print("Test set ready!")
    print ("Data preparation is complete !")

if __name__ == '__main__':
    main(sys.argv[1:])