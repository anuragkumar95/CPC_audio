import os
import glob
import argparse
import sys
from shutil import copyfile
import tqdm
import buckeye
#import torchaudio

def processDataset(audioFiles, outPath, phonesDict):
    """
    List audio files and transcripts for a certain partition of BUCKEYE dataset.
    Args:
        rawPath (string): Directory of BUCKEYE.
        outPath (string): Directory to save BUCKEYE formatted for CPC pipeline
        split (string): Which of the subset of data to take. Either 'train' or 'test'.
    """
    fileWriter = open(os.path.join(outPath, 'converted_aligned_phones.txt'), "a")
    for wavFile in tqdm.tqdm(audioFiles):
        trackName = os.path.basename(wavFile)
        considered_track = buckeye.Track(name=trackName,
                                        words=trackName+ '.words',
                                        phones=trackName + '.phones',
                                        log=trackName + '.log',
                                        txt=trackName + '.txt',
                                        wav=trackName + '.wav')
        fileWriter.write(trackName[3:] + '-')
        for phone in considered_track.phones:
            phoneCode, t0, t1 = phone.seg, phone.beg, phone.end
            phoneCode = str(phonesDict[phoneCode])
            phoneDuration = int((int(t1) - int(t0)) * 100 / 16000)
            fileWriter.write((' ' + phoneCode) * phoneDuration)
        fileWriter.write('\n')
    fileWriter.close()


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
    # sadly no lookup file available
    phones = "aa ae ay aw ao oy ow eh ey er ah uw uh ih iy m n en \
            ng l el t d ch jh th dh sh zh s z k g p b f v w hh y r \
            dx nx tq er em Vn"

    phonesDict = {i+1: j for i, j in enumerate(phones.split())}
    print(phonesDict)
    
    audioFilesTrain = glob.glob(os.path.join(args.pathDB, "speakers/**/*.wav"), recursive=True)
    print(audioFilesTrain)
    #open(os.path.join(args.pathOut, 'converted_aligned_phones.txt'), "w").close()
    #print("Creating data set")
    #processDataset(audioFilesTrain, args.pathOut, 'train', phonesDict)
    print("Data set ready!")
    
    # audioFilesTest = glob.glob(os.path.join(args.pathDB, "TEST/**/*.WAV"), recursive=True)
    # print("Creating test set")
    # processDataset(audioFilesTest, args.pathOut, 'test', phonesDict)
    # print("Test set ready!")
    # print ("Data preparation is complete !")

if __name__ == '__main__':
    main(sys.argv[1:])