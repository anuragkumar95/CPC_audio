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
import pandas as pd
import re

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
                word = multi_laughed_words[0].lower().strip(">") # single word - can recover
            else:
                word = "<LAUGH + multiple words>" # multiple words - can't recover
    return word


def processDataset(audioFiles, 
                    # count_so_far=0, 
                    # words_so_far=None, 
                    labels=None,
                    word_df_so_far=None):
    """
    List audio files and transcripts for a certain partition of TIMIT dataset.
    Args:
        rawPath (string): Directory of TIMIT.
        outPath (string): Directory to save TIMIT formatted for CPC pipeline
        split (string): Which of the subset of data to take. Either 'train' or 'test'.
    """
    # word_count = count_so_far
    # words = [] if words_so_far is None else words_so_far
    word_df = []
    word_labels = {} if labels is None else labels
    for wavFile in tqdm.tqdm(audioFiles):
        print(f"Currently working on {wavFile}")
        current_track = buckeye.Track(name=wavFile[:-4],
                            words=wavFile[:-4] + '.words',
                            phones=wavFile[:-4] + '.phones',
                            log=wavFile[:-4] + '.log',
                            txt=wavFile[:-4] + '.txt')
        for w in current_track.words:
            try:
                word = w.orthography 
            except:
                word = w.entry

            # word_df_fixed_nonwords_words_only.csv
            word = non_word_fixer(word)
            if word == '' or word[0] in ["<", "{"]:
                continue
            
            # word_df_fixed_nonwords.csv
            # word = non_word_fixer(word)
            # if word[:4].upper() == "<EXC":
            #     pass
            else:
                try:
                    temp = word_labels[word]
                except:
                    word_labels[word] = len(word_labels.keys())
                word_df.append({
                    'word': word,
                    'label': word_labels[word]
                })
                
    
    word_df = pd.DataFrame(word_df)
    if word_df_so_far is not None:
        word_df = pd.concat([word_df_so_far, word_df])
        word_df.to_csv("word_df_fixed_nonwords_words_only.csv")
    print(f"Found {word_df['word'].nunique()} words so far")
    return word_df, word_labels
            

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
    args.pathDB = "/pio/scratch/1/i325922/data/BUCKEYE/raw"
    ext = ".wav"
    audioFilesTrain = glob.glob(os.path.join(args.pathDB, "TRAIN/**/*" + ext), recursive=True)
    print("Counting the train set")
    wdf, labs = processDataset(audioFilesTrain)
    
    audioFilesTest = glob.glob(os.path.join(args.pathDB, "TEST/**/*" + ext), recursive=True)
    print("Counting test set")
    wc, ws = processDataset(audioFilesTest, labs, wdf)
    print ("Word count is complete !")

if __name__ == '__main__':
    main(sys.argv[1:])

    