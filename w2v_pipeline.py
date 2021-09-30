import os
import pathlib
import glob
import tqdm
from argparse import ArgumentParser
from collections import defaultdict
from itertools import groupby

from progressbar.bar import DefaultFdMixin
import torch
from gensim.models import Word2Vec
import numpy as np
# import editdistance
from progressbar import ProgressBar
import numpy as np
import pickle
from sklearn.cluster import KMeans
from time import time
import pandas as pd

def ensure_path(path):
    folderpath = (pathlib.Path(path) / '..').resolve()
    if not os.path.exists(folderpath):
        os.makedirs(folderpath)
    return os.path.exists(path)

def cluster_kmeans(data, weights, path, n_clusters, train=True, cosine=False):
    labels = None
    if cosine:
        length = np.sqrt((data**2).sum(axis=1))[:,None]
        length[length <= 0] = 0.00000001 # sometimes these vectors might be zero and following line would crash
        data = data / length

    if not ensure_path(path):
        os.makedirs(path, exist_ok=True)
        if not train:
            raise Exception(f"Tried to cluster data, but there is no kmeans model at {path}. Maybe set train=True?")
        # run k-means
        print("Running kmeans...", flush=True)
        kmeans = KMeans(n_clusters=n_clusters, verbose=1).fit(data, sample_weight=weights)
        with open(path + '/kmeans.pickle', 'wb') as handle:
            pickle.dump(kmeans, handle, protocol=pickle.HIGHEST_PROTOCOL)
        labels = kmeans.labels_
    else:
        with open(path + '/kmeans.pickle', 'rb') as handle:
            kmeans = pickle.load(handle)
        labels = kmeans.predict(data)
        
    return labels


def encode_and_format(segmentation, w2v):
    d = dict()
    q = 0
    res = np.zeros(w2v.vectors.shape)
    cnt = np.ones((w2v.vectors.shape[0]), dtype='int32')
    for sentence in segmentation:
        for word in sentence:
            if word in d.keys():
                cnt[d[word]] += 1
            else:
                d[word] = q
                res[q, :] = w2v[word]
                q += 1

    def r(labels):
        nonlocal d, segmentation
        return [[labels[d[word]] for word in sentence] for sentence in segmentation]

    def m(labels):
        nonlocal d
        return { word: labels[d[word]] for word in d.keys() }

    return res, cnt, r, m


def find_closest_encodings(segmentation, w2v):
    words_list = set(w for sentence in segmentation for w in sentence)
    dictionary = list(w2v.vocab)
    d = {}
    q = 0

    res = np.zeros((len(words_list), w2v.vectors.shape[1]))
    cnt = np.ones((len(words_list)), dtype='int32')
    print("Evaluating semantic vectors...", flush=True)
    bar = ProgressBar(maxval=len(words_list))
    bar.start()

    for i, word in enumerate(words_list):
        bar.update(i)
        if word in d.keys():
            cnt[d[word]] += 1
        else:
            ds = [(editdistance.eval(word, w), w) for w in dictionary]
            m, _ = min(ds)
            candidates = [w for (d0, w) in ds if d0 == m]
            v = np.zeros(w2v.vectors.shape[1])
            
            for c in candidates:
                v += w2v[c]
            
            v /= len(candidates)
            d[word] = q
            res[q, :] = v
            q += 1
    bar.finish()

    def r(labels):
        nonlocal d, segmentation
        return [[labels[d[word]] for word in sentence] for sentence in segmentation]

    def m(labels):
        nonlocal d
        return { word: labels[d[word]] for word in d.keys() }

    return res, cnt, r, m


def vectorize(data, path, vector_size=100, window=5, n_epochs=30, train=True):
    """Run word2vec on the given data.

    Params:

    data: list of sentences (lists of words/strings)
    path: location of the w2v model
    size: size of the model
    """
    if not ensure_path(path):
        os.makedirs(path, exist_ok=True)
        if not train:
            raise Exception(f"Tried to eval word2vec, but there is no model at {path}. Maybe set train=True?")
        # train word2vec
        print("Training word2vec model...", flush=True)
        model = Word2Vec(min_count=1, window=window, vector_size=vector_size)
        model.build_vocab(corpus_iterable=data)
        model.train(corpus_iterable=data, total_examples=model.corpus_count, 
                    total_words=model.corpus_total_words, epochs=n_epochs)
        model.save(str(path + f'/w2v.model'))
    
    model = Word2Vec.load(str(path + f'/w2v.model'))
    if train:
        return encode_and_format(data, model.wv)
    else:
        return find_closest_encodings(data, model.wv)


def parseArgs():
    parser = ArgumentParser()

    parser.add_argument('--pathEncodings', type=str,
                        help='Path to the directory containing word encodings')
    parser.add_argument('--pathWordLabels', type=str,
                        help='Path to a .txt file containing the word labels in order as they '
                        'appear in the audio files')
    parser.add_argument('--pathOut', type=str,
                        help='Path to the directory containing the models, '
                        'if not specified/empty then it will be computed')
    parser.add_argument('--encExtension', type=str, default='.txt',
                        help='Extension of the files containing word encodings')
    parser.add_argument('--word2vec_size', type=int, default=100,
                        help='Size of the word2vec vectors, 100 by default')
    parser.add_argument('--K', type=int, default=10000,
                        help='Number of clusters in KMeans')
    parser.add_argument('--nEpoch', type=int, default=30,
                       help='Number of epoch to run word2vec for')
    parser.add_argument('--windowSize', type=int, default=5,
                       help='Maximum distance between current and predicted word within a sentence in word2vec')
    parser.add_argument('--wordWeights', action='store_true',
                       help='Maximum distance between current and predicted word within a sentence in word2vec')
    parser.add_argument('--seed', type=int, default=290956,
                        help='Random seed')
    return parser.parse_args()

def run(args):

    # TRUE TEXT 
    # df = pd.read_csv(args.pathWordDF)
    # trueWords = df['label']
    # trueWords = list(trueWords.values)


    # LOADING ENCODINGS
    encFiles = glob.glob(os.path.join(args.pathEncodings, '*' + args.encExtension))
    encodings = []
    if args.encExtension == ".txt":
        for encfile in tqdm.tqdm(encFiles):
            with open(encfile, 'r') as f:
                lines = f.readlines()
            for line in lines:
                data = line.split()
                encodings.append([float(x) for x in data])
        wordEncs = torch.tensor(encodings)
    elif args.encExtension == ".pt":
        for encfile in tqdm.tqdm(encFiles):
             encodings.append(torch.tensor(torch.load(encfile)))
        wordEncs = torch.cat(encodings)
    else:
        raise NotImplementedError

    # KMEANS
    with open(args.pathWordLabels, 'r') as f:
        lines = f.readlines()
    allLabels = []
    for line in lines:
        data = line.split()
        recordingLabels = [int(x) for x in data[1:]]
        allLabels += recordingLabels
    uniqueConsecutiveWords = np.array([x[0] for x in groupby(allLabels)])
    unique, counts = np.unique(uniqueConsecutiveWords, return_counts=True)
    weightsDict = dict(zip(unique, counts / uniqueConsecutiveWords.shape[0]))

    if args.wordWeights:
        weights = np.vectorize(weightsDict.get)(uniqueConsecutiveWords)
    else:
        weights = None
    start_time = time()
    kmeans_path = os.path.join(args.pathOut, f"kmeans_K_{args.K}_weighted_{args.wordWeights}")
    wordLabels = cluster_kmeans(wordEncs, weights, kmeans_path, n_clusters=args.K, cosine=True)
    print(f"Finished clustering in {time() - start_time} seconds")
    hexedWordLabels = list(map(hex, wordLabels))

    n_words_in_sentence = 15
    splitHexLabels = []
    for i in range(len(hexedWordLabels) // n_words_in_sentence):
        sentence = hexedWordLabels[i * n_words_in_sentence : (i+1) * n_words_in_sentence]
        splitHexLabels.append(sentence)
    splitHexLabels.append(hexedWordLabels[(i+1) * n_words_in_sentence:])
    print(f"Original quantized corpus of length {len(hexedWordLabels)} successfully split into {(len(hexedWordLabels) // n_words_in_sentence) + 1} sentences!")

    # W2V
    word_time = time()
    word2vec_path = os.path.join(args.pathOut, f"word2vec_window_{args.windowSize}_nepoch_{args.nEpoch}_vectorsize_{args.word2vec_size}_K{args.K}_weighted_{args.wordWeights}")
    encodings, weights, reconstruct, build_map = vectorize(splitHexLabels, word2vec_path, args.word2vec_size, 
                                                            window=args.windowSize, n_epochs=args.nEpoch)
    # for w in range(5, 20):
    #     word2vec_path = os.path.join(args.pathOut, f"word2vec_window_{w}")
    #     encodings, weights, reconstruct, build_map = vectorize(wordLabels, word2vec_path, args.word2vec_size, window=w)

    print(f"Finished word2vec in {time() - word_time} seconds")
    print('Done!')
    
if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    args = parseArgs()
    run(args)