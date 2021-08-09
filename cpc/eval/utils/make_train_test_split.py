import os
import glob
import argparse
import sys
import tqdm

def create_split_file(audioFiles, outPath, split):
    fileWriter = open(os.path.join(outPath, f'{split}_split.txt'), "a")
    for wavFile in tqdm.tqdm(audioFiles):
        speakerName = os.path.basename(os.path.dirname(wavFile))
        trackName = os.path.basename(wavFile)
        fileWriter.write(trackName[:-4])
        #fileWriter.write(trackName[:-4])
        fileWriter.write('\n')
    fileWriter.close()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Create .txt files with train-test split of Buckeye data.')
    parser.add_argument('--pathDB', type=str,
                        help='Path to the directory containing the audio '
                        'files')
    parser.add_argument("--pathOut", type=str,
                        help='Path out the output directory')
    return parser.parse_args(argv)

def main(argv):

    args = parse_args(argv)
    os.makedirs(args.pathOut, exist_ok=True)

    audioFiles = glob.glob(os.path.join(args.pathDB, "TRAIN/**/*.wav"), recursive=True)
    open(os.path.join(args.pathOut, 'train_split.txt'), "w").close()
    create_split_file(audioFiles, args.pathOut, 'train')

    audioFiles = glob.glob(os.path.join(args.pathDB, "TEST/**/*.wav"), recursive=True)
    open(os.path.join(args.pathOut, 'test_split.txt'), "w").close()
    create_split_file(audioFiles, args.pathOut, 'test')


if __name__ == '__main__':
    main(sys.argv[1:])