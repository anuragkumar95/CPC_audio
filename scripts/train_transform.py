#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from logging import raiseExceptions
import os
import sys
import tqdm

import h5py
import numpy as np
from scipy.linalg import eigh
from scipy.sparse import coo_matrix

def read_vectors_from_txt_file(path_to_files):
    """ Read vectors in text format from path_to_files.

        Yields:
            Tuple(str, np.array): utt and vector
    """
    #ret_val = {}
    print("Reading input embeddings.")
    for file in tqdm.tqdm(os.listdir(path_to_files)):
        if file.split('.')[-1] != 'txt':
            continue
        feats = np.loadtxt(os.path.join(path_to_files, file))
        #ret_val[file.split('.')[0]] = feats
        if len(file.split('.')[0]) <= 6:
            print(file.split('.')[0])
        yield file.split('.')[0], feats

def read_txt_vectors_from_stdin():
    """ Read vectors in text format. This code expects correct format.

        Yields:
            Tuple(str, np.array): utt and vector
    """
    for line in sys.stdin:
        splitted_line = line.split()
        name = splitted_line[0]
        end_idx = splitted_line.index(']')
        vector_data = np.array([float(single_float) for single_float in splitted_line[2:end_idx]])
        yield name, vector_data


def write_txt_vector_to_stdout(utt, emb):
    """ Write vectors file in text format.

    Args:
        utt (str):
        emb (np.ndarray):
    """
    sys.stdout.write('{}  [ {} ]{}'.format(utt, ' '.join(str(x) for x in emb), os.linesep))


def l2_norm(vec_or_matrix):
    """ L2 normalization of vector array.

    Args:
        vec_or_matrix (np.array): one vector or array of vectors

    Returns:
        np.array: normalized vector or array of normalized vectors
    """
    if len(vec_or_matrix.shape) == 1:
        # linear vector
        return vec_or_matrix / np.linalg.norm(vec_or_matrix)
    elif len(vec_or_matrix.shape) == 2:
        return vec_or_matrix / np.linalg.norm(vec_or_matrix, axis=1, ord=2)[:, np.newaxis]
    else:
        raise ValueError('Wrong number of dimensions, 1 or 2 is supported, not %i.' % len(vec_or_matrix.shape))


def get_class_means_between_and_within_covs(samples, classids, bias=True):
    """ Return class means and the between and shread within class covariance matrix.

    Args:
        samples (np.array): input data
        classids (np.array): identification of classes
        bias (bool): type of normalization, see np.cov for more information

    Returns:
        tuple: class means, between class covariance, within class covariance
    """
    nsamples, dim = samples.shape
    sample2class = classids_to_posteriors(classids)
    counts = np.array(sample2class.sum(1))
    means = np.array(sample2class.dot(samples)) / counts
    between_cov = (means - samples.mean(0)) * np.sqrt(counts)
    between_cov = between_cov.T.dot(between_cov) / nsamples
    within_cov = np.cov(samples.T, bias=bias) - between_cov
    return means, between_cov, within_cov


def classids_to_posteriors(classids, dtype='f4'):
    """ Transform classids into a sparse matrix of 1 and 0 that can be interpreted as posteriors.

    Args:
        classids (np.array): class identifications
        dtype (str): dtype o output sparse matrix

    Returns:
        scipy.sparse.coo_matrix: sparse matrix of 1 and 0 that can be interpreted as posteriors
    """
    nsamples = np.array(classids).squeeze().shape[0]
    class2post = coo_matrix((np.ones(nsamples, dtype), (classids, list(range(nsamples))))).tocsr()
    return class2post


def parse_align(alignment, path_to_files):
    """
    Function to parse earnings21 alignment files.
    Args: 
        alignment: path to file which containes spk aligned reference. 
        path_to_files: path to earnings21 wav files.
    Returns:
        features for the corresponding file.
        dictionary for spk to feature index mapping. 
    """
    with open(alignment, "r") as f:
        lines = f.readlines()
        lines = [l.strip().split('|') for l in lines]
        spk_dur = {l[1]:[] for l in lines[1:]}
        i = 1
        while(i < len(lines) and lines[i][2] == ''):
            i+=1
        if i == len(lines):
            return [], []
        cur_spk = lines[i][1]
        start = float(lines[i][2])
        end = 0
        for i, line in enumerate(lines[i+1:]):
            spk = line[1]
            if spk != cur_spk:
                if i > 0:
                    spk_dur[cur_spk].append((start, end))
                cur_spk = spk
                start = line[2]
                if len(start) > 0:
                    start = float(start)
            else:
                end = line[3]
                if len(end) > 0:
                    end = float(end)
        
        spk_2_indx = {l[1]:[] for l in lines[1:]}
        for spk in spk_dur:
            for start, end in spk_dur[spk]:
                try:
                    start_idx = int(start*100)
                    end_idx = int(end*100)
                    spk_2_indx[spk].append((start_idx, end_idx))
                except:
                    continue
    
    file = alignment.split('/')[-1].split('.')[0] + '.txt'
    feats = np.loadtxt(os.path.join(path_to_files, file))
    return feats, spk_2_indx


def train(embeddings, labels, lda_dim, whiten):
    n_vectors, vector_dim = embeddings.shape

    unique_idxs = list(set(labels))
    idxs = np.array([unique_idxs.index(x) for x in labels])
    # subtract mean
    mean1 = np.mean(embeddings, axis=0)
    embeddings -= mean1

    # l2-norm
    embeddings = l2_norm(embeddings)

    # train LDA
    means, between_cov, within_cov = get_class_means_between_and_within_covs(embeddings, idxs)

    # Hossein's magic - within_cov has probably smaller numbers on training data than within_cov on test data
    # CW = CW + 0.01 * eye(size(CT, 1));
    within_cov = within_cov + 0.01 * np.diag(np.ones(within_cov.shape[0]))

    w, v = eigh(between_cov, within_cov)
    index = np.argsort(w)[::-1]
    v = v[:, index]

    lda = v[:, 0:lda_dim]
    if whiten:
        lda = lda.dot(np.diag(1. / np.sqrt(np.diag(lda.T.dot(between_cov + within_cov).dot(lda)))))

    embeddings = embeddings.dot(lda)

    # subtract mean again
    mean2 = np.mean(embeddings, axis=0)
    embeddings -= mean2

    # l2-norm again
    embeddings = l2_norm(embeddings)

    return embeddings, mean1, lda, mean2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--utt2spk', required=True, type=str, help='path to spk2utt')
    parser.add_argument('--outputs', required=False, type=str, help='path to embeddings stored in text format')
    parser.add_argument('--output-h5', required=True, type=str, help='path to output h5 file')
    parser.add_argument('--lda-dim', required=False, default=128, type=int, help='LDA dimensionality')
    parser.add_argument('--whiten', required=False, default=False, action='store_true', help='do whitenning after LDA')
    parser.add_argument('--mode', required=False, default=None, help='do LDA on mean of embeddings or random embeddings.')
    parser.add_argument('--db', required=False, type=str, help='Argument to handle earnings21 data')
    parser.add_argument('--alignment', required=False, type=str, help='Path to Earnings21 aligned files.')
    parser.add_argument('--save', required=False, type=str, help='Path to save dir for LDA embeddings.')
    
    args = parser.parse_args()
    
    if args.db == 'earnings21':
        for align in tqdm.tqdm(os.listdir(args.alignment)):
            labels = []
            embeddings = []
            if 'normalized' in align:
                continue
            feats, spk2idx = parse_align(os.path.join(args.alignment, align), args.outputs)
            if len(feats) == 0:
                continue
            for spk in spk2idx:
                for start, end in spk2idx[spk]:
                    if start < end:
                        embedding = np.mean(feats[start:end], axis = 0)
                        embeddings.append(embedding)
                        labels.append(spk)

    else:
        utt2spk = {}
        with open(args.utt2spk) as f:
            for line in f:
                utt, spk = line.split()
                utt2spk[utt] = spk
        
        embeddings, labels, utts = [], [], []
        
        if not args.outputs:   
            for utt, emb in read_txt_vectors_from_stdin():
                try:
                    labels.append(utt2spk[utt])
                    utts.append(utt)
                    embeddings.append(emb)
                except KeyError:
                    pass
        else:
            #utt2feats = read_vectors_from_txt_file(args.outputs)
            for utt, feats in read_vectors_from_txt_file(args.outputs):
                if args.mode == 'mean':
                    labels.append(utt2spk[utt])
                    mean = np.mean(feats, axis=0)
                    embeddings.append(mean)
                    utts.append(utt) 
                elif args.mode == 'random':
                    np.random.seed(123)
                    ind = np.random.choice(feats.shape[0], 5)
                    for i in ind:
                        embeddings.append(feats[i])
                        labels.append(utt2spk[utt])
                    utts.append(utt)
                else:
                    for frame in feats:
                        labels.append(utt2spk[utt])
                        embeddings.append(frame)
                    utts.append(utt)
        
        #embeddings, mean1, lda, mean2 = train(np.array(embeddings), labels, lda_dim=args.lda_dim, whiten=args.whiten)

    embeddings, mean1, lda, mean2 = train(np.array(embeddings), labels, lda_dim=args.lda_dim, whiten=args.whiten)
            
    # dump them into h5 file
    with h5py.File(args.output_h5, 'w') as f:
        f.create_dataset('mean1', data=mean1)
        f.create_dataset('mean2', data=mean2)
        f.create_dataset('lda', data=lda)

    if args.save:
        if not os.path.isdir(args.save):
            os.makedirs(args.save)
        save_path = os.path.join(args.save, align.split('.')[0]+'.txt')
        np.savetxt(save_path, embeddings)
    
    else:
        for utt, embedding in zip(utts, embeddings):
            write_txt_vector_to_stdout(utt, embedding)
    
    
    
    # train parameters
    