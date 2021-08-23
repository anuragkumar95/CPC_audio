# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import json
import numpy as np
import random
import torch
import sys
import psutil
from copy import deepcopy
from bisect import bisect_left
import torch.nn.functional as F
from scipy.signal import find_peaks
# import matplotlib.pyplot as plt

def maxMinNorm(x):
    x -= x.min(-1, keepdim=True)[0]
    x /= x.max(-1, keepdim=True)[0]
    return x

def levenshteinDistance(data):
    s1, s2 = data
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def getAverageSlices(peaks, originalLength, device, minLengthSeq=None):
    # now work out cut indices in each minibatch element
    # batch_elem_idx = idx // encodedData.size(1)
    # transition_idx = F.pad(torch.nonzero(batch_elem_idx[1:] != batch_elem_idx[:-1]), (0,0, 1,0))
    cutpoints = torch.nonzero((peaks % originalLength) == 0)
    compressedLens = (cutpoints[1:]-cutpoints[:-1]).squeeze(1)
    # Handling case when there are sequences shorter than nPredict + 1
    tooShortBatchIdx = torch.where(compressedLens < minLengthSeq)[0]
    if len(tooShortBatchIdx) > 0:
        for i in tooShortBatchIdx:
            # How many sequence elements are we missing to be able to predict
            cuts2Add = minLengthSeq - compressedLens[i]
            # We add them by splitting the largest segments in the sequence in two equal parts
            for j in range(cuts2Add):
                segmentsBoundaries = peaks[cutpoints[i]:cutpoints[i + 1] + 1 + j]
                largestSegmentIdx = torch.diff(segmentsBoundaries).max(0)[1]
                peaks = torch.cat((peaks[:cutpoints[i] + largestSegmentIdx + 1], 
                                torch.round((segmentsBoundaries[largestSegmentIdx] + segmentsBoundaries[largestSegmentIdx + 1]) / 2).int().unsqueeze(0), 
                                peaks[cutpoints[i] + largestSegmentIdx + 1:]))
            cutpoints[i + 1:] += cuts2Add
        cutpoints = torch.nonzero((peaks % originalLength) == 0)
        compressedLens = (cutpoints[1:]-cutpoints[:-1]).squeeze(1)

    seqIdx = torch.nn.utils.rnn.pad_sequence(
        torch.split(peaks[1:] % originalLength, tuple(cutpoints[1:]-cutpoints[:-1])), batch_first=True)
    seqIdx[seqIdx==0] = originalLength
    seqIdx = F.pad(seqIdx, (1,0,0,0))

    frame_idxs = torch.arange(originalLength, device=device).view(1, 1, -1)
    compressMatrices = (
        (seqIdx[:,:-1, None] <= frame_idxs)
        & (seqIdx[:,1:, None] > frame_idxs)
    ).float()

    compressedLens = compressedLens.cpu()
    return compressMatrices, compressedLens

def kreukBoundaryDetector(encodedData, prominence, minLengthSeq=None):
    scores = F.cosine_similarity(encodedData[:, :-1, :], encodedData[:, 1:, :], dim=-1)
    scores = torch.cat([scores[:, 0].view(-1, 1), scores], dim=1)
    scores = 1 - maxMinNorm(scores)
    peaks, _ = find_peaks(scores.view(-1).cpu().numpy(), prominence=prominence)
    if len(peaks) == 0:
        peaks = torch.tensor([0])
    peaks = torch.tensor(peaks)
    # Ensure that minibatch boundaries are preserved
    seqEndIdx = torch.arange(0, encodedData.size(0)*encodedData.size(1) + 1, encodedData.size(1))
    peaks = torch.unique(torch.cat((peaks, seqEndIdx)), sorted=True)
    return peaks

def jchBoundaryDetector(encodedData, final_length_factor, minLengthSeq=None, step_reduction=0.2):
    assert not torch.isnan(encodedData).any()
    device = encodedData.device
    encFlat = F.pad(encodedData.reshape(-1, encodedData.size(-1)).detach(), (0, 0, 1, 0))
    feat_csum = encFlat.cumsum(0)
    feat_csum2 = (encFlat**2).cumsum(0)
    peaks = torch.arange(feat_csum.size(0), device=feat_csum.device)
    final_length = int(final_length_factor * len(encFlat))
    while len(peaks) > final_length:
        begs = peaks[:-2]
        ends = peaks[2:]
        sum1 = (feat_csum.index_select(0, ends) - feat_csum.index_select(0, begs))
        sum2 = (feat_csum2.index_select(0, ends) - feat_csum2.index_select(0, begs))
        num_elem = (ends-begs).float().unsqueeze(1)

        diffs = F.pad(torch.sqrt(((sum2/ num_elem - (sum1/ num_elem)**2) ).mean(1)) * num_elem.squeeze(1),
                    (1,1), value=1e10)

        num_to_retain = max(final_length, int(peaks.shape[-1] * step_reduction))
        _, keep_idx = torch.topk(diffs, num_to_retain)
        keep_idx = torch.sort(keep_idx)[0]
        peaks = peaks.index_select(0, keep_idx)
    # Ensure that minibatch boundaries are preserved
    seq_end_idx = torch.arange(0, encodedData.size(0)*encodedData.size(1) + 1, encodedData.size(1), device=device)
    peaks = torch.unique(torch.cat((peaks, seq_end_idx)), sorted=True)
    return peaks

def jhuBoundaryDetector(encodedData, threshold=0.04):
    # BS x Len x DimEnc
    device = encodedData.device
    batchSize, Len, _ = encodedData.shape
    d_s = F.cosine_similarity(encodedData[:, :-1, :], encodedData[:, 1:, :], dim=-1)
    dsmin = torch.min(d_s, dim=-1)[0].view(-1, 1)
    dsmax = torch.max(d_s, dim=-1)[0].view(-1, 1)
    d = 1 - (d_s - dsmin) / (dsmax - dsmin)

    zeros_2_comp = torch.zeros_like(d, device=device)
    pt_1 = torch.minimum(torch.maximum(F.pad(d[:,1:] - d[:,:-1], (1,0)), zeros_2_comp), 
                            torch.maximum(F.pad(d[:,:-1] - d[:,1:], (0,1)), zeros_2_comp))
    pt_2 = torch.minimum(torch.maximum(F.pad(d[:,2:] - d[:,:-2], (2,0)), zeros_2_comp), 
                            torch.maximum(F.pad(d[:,:-2] - d[:,2:], (0,2)), zeros_2_comp))
    p = torch.minimum(torch.maximum(torch.maximum(pt_1, pt_2) - threshold, zeros_2_comp), pt_1)
    p = F.pad(p, (1,0), value=1)

    b_soft = torch.tanh(10 * p)
    b_hard = torch.tanh(10000 * p)
    b = b_soft + (b_hard - b_soft).detach() # stopping gradient in PyTorch?
    

    compressed_lens = torch.sum(b, dim=1).int().cpu()
    M = torch.max(compressed_lens).int()
    # very ugly?
    U = torch.arange(1, M+1, device=device).view(M, -1).expand((batchSize, -1, Len))
    ret = torch.nn.utils.rnn.pack_padded_sequence(U, compressed_lens,
                                 batch_first=True, enforce_sorted=False)
    U = torch.nn.utils.rnn.pad_packed_sequence(ret, batch_first=True)[0]

    V = U.permute(0, 2, 1) - torch.cumsum(b, dim=1).view(batchSize, Len, 1)
    W = 1 - torch.tanh(100000 * abs(V))
    W /= torch.maximum(torch.sum(W, dim=1).view(batchSize, 1, M), torch.ones(batchSize, 1, M, device=device))
    W = W.permute(0, 2, 1)
    ZW = W @ encodedData
    
    return ZW, W, compressed_lens

def compress_batch(encodedData, compress_matrices, compressed_lens, pack=False):
    ret = torch.bmm(
        compress_matrices / torch.maximum(compress_matrices.sum(-1, keepdim=True), torch.ones(1, device=compress_matrices.device)), 
        encodedData)
    if pack:
        ret = torch.nn.utils.rnn.pack_padded_sequence(ret, compressed_lens, batch_first=True, enforce_sorted=False)
    return ret


def decompress_padded_batch(compressed_data, compress_matrices):
    if isinstance(compressed_data, torch.nn.utils.rnn.PackedSequence):
        # We pad to have the maximum possible sequence length so as to be compatible with multi GPU setup
        compressed_data, _ = torch.nn.utils.rnn.pad_packed_sequence(
            compressed_data, batch_first=True, total_length=compress_matrices.size(2))
    #assert (compress_matrices.sum(1) == 1).all()
    return compressed_data


def seDistancesToCentroids(vecs, centroids, doNorm=False):
    
    if len(vecs.shape) == 2:
        vecs = vecs.view(1, *(vecs.shape))

    B = vecs.shape[0]
    N = vecs.shape[1]
    k = centroids.shape[0]

    # vecs: B x L x Dim
    # centroids: k x Dim

    if doNorm:
        vecLengths = torch.sqrt((vecs*vecs).sum(-1))
        vecs = vecs / vecLengths.view(B, N, 1)
        centrLengths = torch.sqrt((centroids*centroids).sum(-1))
        centroids = centroids / centrLengths.view(k, 1)
        
    return torch.square(centroids).sum(1).view(1, 1, -1) + torch.square(vecs).sum(-1).view(B, N, 1) \
        - 2*(vecs.view(B, N, 1, -1) * centroids.view(1, 1, k, -1)).sum(-1)  #torch.matmul(vecs, centroids.T)


def pushToClosestForBatch(points, centers, deg=0.5, doNorm=False, doNormForPush=False):

    B = points.shape[0]   
    N = points.shape[1]
    k = centers.shape[0]

    if doNormForPush:
        pointsLengths = torch.sqrt((points*points).sum(-1))
        points = points / pointsLengths.view(B, N, 1)
        centrLengths = torch.sqrt((centers*centers).sum(-1))
        centers = centers / centrLengths.view(k, 1)

    distsSq = seDistancesToCentroids(points, centers, doNorm=doNorm)
    dists = torch.sqrt(distsSq)
     
    closest = dists.argmin(-1)
    diffs = centers[closest].view(B, N, -1) - points
    res = deg * diffs + points
     
    return res


def untensor(d):
    if isinstance(d, list):
        return [untensor(v) for v in d]
    if isinstance(d, dict):
        return dict((k, untensor(v)) for k, v in d.items())
    if hasattr(d, 'tolist'):
        return d.tolist()
    return d


def save_logs(data, pathLogs):
    with open(pathLogs, 'w') as file:
        json.dump(data, file, indent=2)


def update_logs(logs, logStep, prevlogs=None):
    out = {}
    for key in logs:
        out[key] = deepcopy(logs[key])

        if prevlogs is not None:
            out[key] -= prevlogs[key]
        out[key] /= logStep
    return out


def show_logs(text, logs):
    print("")
    print('-'*50)
    print(text)

    for key in logs:

        if key == "iter":
            continue

        nPredicts = logs[key].shape[0]

        strSteps = ['Step'] + [str(s) for s in range(1, nPredicts + 1)]
        formatCommand = ' '.join(['{:>16}' for x in range(nPredicts + 1)])
        print(formatCommand.format(*strSteps))

        strLog = [key] + ["{:10.6f}".format(s) for s in logs[key]]
        print(formatCommand.format(*strLog))

    print('-'*50)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cpu_stats():
    print(sys.version)
    print(psutil.cpu_percent())
    print(psutil.virtual_memory())


def ramp_scheduling_function(n_epoch_ramp, epoch):
    if epoch >= n_epoch_ramp:
        return 1
    else:
        return (epoch + 1) / n_epoch_ramp


class SchedulerCombiner:
    r"""
    An object which applies a list of learning rate schedulers sequentially.
    """

    def __init__(self, scheduler_list, activation_step, curr_step=0):
        r"""
        Args:
            - scheduler_list (list): a list of learning rate schedulers
            - activation_step (list): a list of int. activation_step[i]
            indicates at which step scheduler_list[i] should be activated
            - curr_step (int): the starting step. Must be lower than
            activation_step[0]
        """

        if len(scheduler_list) != len(activation_step):
            raise ValueError("The number of scheduler must be the same as "
                             "the number of activation step")
        if activation_step[0] > curr_step:
            raise ValueError("The first activation step cannot be higher than "
                             "the current step.")
        self.scheduler_list = scheduler_list
        self.activation_step = deepcopy(activation_step)
        self.curr_step = curr_step

    def step(self):
        self.curr_step += 1
        index = bisect_left(self.activation_step, self.curr_step) - 1
        for i in reversed(range(index, len(self.scheduler_list))):
            self.scheduler_list[i].step()

    def __str__(self):
        out = "SchedulerCombiner \n"
        out += "(\n"
        for index, scheduler in enumerate(self.scheduler_list):
            out += f"({index}) {scheduler.__str__()} \n"
        out += ")\n"
        return out
