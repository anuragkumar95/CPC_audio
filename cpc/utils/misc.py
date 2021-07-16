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

def sequence_segmenter(encodedData, final_length_factor, step_reduction=0.2):
    assert not torch.isnan(encodedData).any()
    device = encodedData.device
    encFlat = F.pad(encodedData.reshape(-1, encodedData.size(-1)).detach(), (0, 0, 1, 0))
    feat_csum = encFlat.cumsum(0)
    feat_csum2 = (encFlat**2).cumsum(0)
    idx = torch.arange(feat_csum.size(0), device=feat_csum.device)

    final_length = int(final_length_factor * len(encFlat))

    while len(idx) > final_length:
        begs = idx[:-2]
        ends = idx[2:]

        sum1 = (feat_csum.index_select(0, ends) - feat_csum.index_select(0, begs))
        sum2 = (feat_csum2.index_select(0, ends) - feat_csum2.index_select(0, begs))
        num_elem = (ends-begs).float().unsqueeze(1)

        diffs = F.pad(torch.sqrt(((sum2/ num_elem - (sum1/ num_elem)**2) ).mean(1)), (1,1), value=1e10)

        num_to_retain = max(final_length, int(idx.shape[-1] * step_reduction))
        _, keep_idx = torch.topk(diffs, num_to_retain)
        keep_idx = torch.sort(keep_idx)[0]
        idx = idx.index_select(0, keep_idx)
    
    # Ensure that minibatch boundaries are preserved
    seq_end_idx = torch.arange(0, encodedData.size(0)*encodedData.size(1), encodedData.size(1), device=device)
    idx = torch.unique(torch.cat((idx, seq_end_idx)), sorted=True)

    # now work out cut indices in each minibatch element
    batch_elem_idx = idx // encodedData.size(1)
    transition_idx = F.pad(torch.nonzero(batch_elem_idx[1:] != batch_elem_idx[:-1]), (0,0, 1,0))
    cutpoints = (torch.nonzero((idx % encodedData.size(1)) == 0))
    compressed_lens = (cutpoints[1:]-cutpoints[:-1]).squeeze(1)

    seq_idx = torch.nn.utils.rnn.pad_sequence(
        torch.split(idx[1:] % encodedData.size(1), tuple(cutpoints[1:]-cutpoints[:-1])), batch_first=True)
    seq_idx[seq_idx==0] = encodedData.size(1)
    seq_idx = F.pad(seq_idx, (1,0,0,0))

    frame_idxs = torch.arange(encodedData.size(1), device=device).view(1, 1, -1)
    compress_matrices = (
        (seq_idx[:,:-1, None] <= frame_idxs)
        & (seq_idx[:,1:, None] > frame_idxs)
    ).float()

    print(f"{final_length} -> {len(idx)}")
    compressed_lens = compressed_lens.cpu()
    try:
        assert compress_matrices.shape[0] == encodedData.shape[0]
    except:
        print(compress_matrices.shape[0])
        print(encodedData.shape[0])
    #     sys.exit(0)
    return compress_matrices, compressed_lens


def compress_batch(encodedData, compress_matrices, compressed_lens):
    ret = torch.bmm(
        compress_matrices / torch.maximum(compress_matrices.sum(-1, keepdim=True), torch.ones(1, device=compress_matrices.device)), 
        encodedData)
    return ret, torch.nn.utils.rnn.pack_padded_sequence(ret, compressed_lens, batch_first=True, enforce_sorted=False)


def decompress_padded_batch(compressed_data, compress_matrices, compressed_lens):
    if isinstance(compressed_data, torch.nn.utils.rnn.PackedSequence):
        compressed_data, unused_lens = torch.nn.utils.rnn.pad_packed_sequence(
            compressed_data, batch_first=True, total_length=compress_matrices.size(1))
    assert (compress_matrices.sum(1) == 1).all()
    return compressed_data
    # return torch.bmm(
    #     compress_matrices.transpose(1, 2), compressed_data)


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
