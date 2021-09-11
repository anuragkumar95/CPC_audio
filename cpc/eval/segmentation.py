# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import sys
import torch
import json
import time
import numpy as np
from pathlib import Path
from copy import deepcopy
import os
import tqdm

import cpc.criterion as cr
import cpc.criterion.soft_align as sa
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
from cpc.model import CPCModelNullspace
# import pandas as pd

def getCriterion(args, downsampling, nSpeakers):
    dimFeatures = args.hiddenGar if not args.onEncoder else args.hiddenEncoder
    if args.cpc_mode == 'none':
        cpcCriterion = cr.NoneCriterion()
    else:
        sizeInputSeq = (args.sizeWindow // downsampling)
        if args.CPCCTC:
            cpcCriterion = sa.CPCUnsupersivedCriterion(args.nPredicts,
                                                    args.CPCCTCNumMatched,
                                                    args.hiddenGar,
                                                    args.hiddenEncoder,
                                                    args.negativeSamplingExt,
                                                    negativeSamplingExt2=args.negativeSamplingExt2,
                                                    nPredicts2=args.nPredicts2,
                                                    nMatched2=args.CPCCTCNumMatched2,
                                                    simMeasure=args.simMeasure,
                                                    allowed_skips_beg=args.CPCCTCSkipBeg,
                                                    allowed_skips_end=args.CPCCTCSkipEnd,
                                                    predict_self_loop=args.CPCCTCSelfLoop,
                                                    learn_blank=args.CPCCTCLearnBlank,
                                                    normalize_enc=args.CPCCTCNormalizeEncs,
                                                    normalize_preds=args.CPCCTCNormalizePreds,
                                                    masq_rules=args.CPCCTCMasq,
                                                    loss_temp=args.CPCCTCLossTemp,
                                                    no_negs_in_match_window=args.CPCCTCNoNegsMatchWin,
                                                    limit_negs_in_batch=args.limitNegsInBatch,
                                                    mode=args.cpc_mode,
                                                    rnnModeRegHead=args.rnnModeRegHead,
                                                    rnnModeDownHead=args.rnnModeDownHead,
                                                    dropout=args.dropout,
                                                    nSpeakers=nSpeakers,
                                                    speakerEmbedding=args.speakerEmbedding,
                                                    sizeInputSeq=sizeInputSeq,
                                                    numLevels=args.CPCCTCNumLevels,
                                                    segmentationThreshold=args.segmentationThreshold,
                                                    smartPooling=args.smartPooling,
                                                    stepReduction=args.stepReduction,
                                                    NoARonRegHead=args.NoARonRegHead)

        else:
            cpcCriterion = cr.CPCUnsupersivedCriterion(args.nPredicts,
                                                    args.hiddenGar,
                                                    args.hiddenEncoder,
                                                    args.negativeSamplingExt,
                                                    simMeasure=args.simMeasure,
                                                    mode=args.cpc_mode,
                                                    rnnModeRegHead=args.rnnModeRegHead,
                                                    rnnModeDownHead=args.rnnModeDownHead,
                                                    dropout=args.dropout,
                                                    nSpeakers=nSpeakers,
                                                    speakerEmbedding=args.speakerEmbedding,
                                                    sizeInputSeq=sizeInputSeq)
    return cpcCriterion

def evalPhoneSegmentation(featureMaker, criterion, boundaryDetector, dataLoader, 
                          onEncodings=True, toleranceInFrames=2, wordSegmentation=False):

    featureMaker.eval()
    logs = {"precision": 0, "recall": 0, "f1": 0, "r": 0}
    EPS = 1e-7
    segmentationParamRange = np.arange(0.0, 0.15, 0.01)

    # results = []
    for step, fulldata in tqdm.tqdm(enumerate(dataLoader)):
        with torch.no_grad():
            batchData, labelData = fulldata
            labelPhones = labelData['phone']
            label = labelData['word'] if wordSegmentation else labelPhones
            cFeature, encodedData, _ = featureMaker(batchData, labelPhones)
            if wordSegmentation:
                cFeature = cFeature[1]
                encodedData = cFeature['encodedData']
                seqLens = cFeature['seqLens'] - 1
                segmentLensBatch = cFeature['segmentLens'].cpu()
                cFeature = cFeature['states']
                cFeature = criterion.module.wPredictions[1].predictors[0](cFeature[:, :-criterion.module.nMatched[1], :]).squeeze()
                features = (cFeature[:, :-1, :], encodedData[:, 1:, :])
            else:
                cFeature = cFeature[0]
                B, S, H = cFeature.size()
                seqLens = torch.ones(B, dtype=torch.int64, device=cFeature.device) * S
                segmentLensBatch = torch.ones(B, dtype=torch.int64, device=cFeature.device)
                features = (encodedData[:, :-1, :], encodedData[:, 1:, :]) if onEncodings else (cFeature[:, :-1, :], cFeature[:, 1:, :])
        maxRval = -np.inf
        diffs = torch.diff(label, dim=1)
        trueBoundariesBatch = []
        for b in range(diffs.size(0)):
            boundaries = torch.nonzero(diffs[b].contiguous().view(-1), as_tuple=True)[0] + 1
            boundaries = torch.cat((torch.Tensor([0]), boundaries))
            trueBoundariesBatch.append(boundaries)

        for segmentationParam in segmentationParamRange:
            predictedBoundariesBatch = boundaryDetector(features, segmentationParam, seqLens)
            
            precisionCounter = 0
            recallCounter = 0
            numPredictedBounds = 0
            numTrueBounds = 0

            for predictedBoundaries, trueBoundaries, segmentLens in zip(predictedBoundariesBatch, trueBoundariesBatch, segmentLensBatch):
                if wordSegmentation:
                    predictedBoundaries = segmentLens[predictedBoundaries]
                numPredictedBounds += len(predictedBoundaries)
                numTrueBounds += len(trueBoundaries)
                for predictedBoundary in predictedBoundaries:
                    minDist = torch.min(torch.abs(trueBoundaries - predictedBoundary))
                    precisionCounter += (minDist <= toleranceInFrames)
                for trueBoundary in trueBoundaries:
                    minDist = torch.min(torch.abs(predictedBoundaries - trueBoundary))
                    detected = (minDist <= toleranceInFrames)
                    recallCounter += detected

            precision = precisionCounter / (numPredictedBounds + EPS)
            recall = recallCounter / (numTrueBounds + EPS)
            f1 = 2 * (precision * recall) / (precision + recall + EPS)
            os = recall / (precision + EPS) - 1
            r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
            r2 = (-os + recall - 1) / (np.sqrt(2))
            rVal = 1 - (np.abs(r1) + np.abs(r2)) / 2
            
            if rVal > maxRval:
                bestSegmentationParam = segmentationParam
                maxRval = rVal
                bestPrecision = precision
                bestRecall = recall
                bestF1 = f1

        logs["precision"] += bestPrecision.view(1).numpy()
        logs["recall"] += bestRecall.view(1).numpy()
        logs["f1"] += bestF1.view(1).numpy()
        logs["r"] += maxRval.view(1).numpy()
    logs = utils.update_logs(logs, step)

    return logs


def run(featureMaker,
        criterion,
        boundaryDetector,
        dataLoader,
        pathCheckpoint,
        onEncodings,
        wordSegmentation=False):
    print("%d batches" % len(dataLoader))
    logs = evalPhoneSegmentation(featureMaker, criterion, boundaryDetector, dataLoader, 
                                 onEncodings, wordSegmentation=wordSegmentation)
    utils.show_logs("Results", logs)
    for key, value in dict(logs).items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logs[key] = value
    utils.save_logs(logs, f"{pathCheckpoint}_logs.json")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Linear separability trainer'
                                     ' (default test in speaker separability)')
    parser.add_argument('--pathDB', type=str, nargs="+",
                        help="Path to the directory containing the audio data.")
    parser.add_argument('--pathTrain', type=str, nargs="+",
                        help="Path to the list of the training sequences.")
    parser.add_argument('--pathVal', type=str, nargs="+",
                        help="Path to the list of the test sequences.")
    parser.add_argument('--load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels.")
    parser.add_argument('--pathWords', type=str, default=None,
                        help="Path to the word labels. If given, will"
                        " compute word separability.")
    parser.add_argument('--pathCheckpoint', type=str, default='out',
                        help="Path of the output directory where the "
                        " checkpoints should be dumped.")
    parser.add_argument('--nGPU', type=int, default=-1,
                        help='Bumber of GPU. Default=-1, use all available '
                        'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--file_extension', type=str, nargs="+", default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")
    parser.add_argument('--size_window', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--nProcessLoader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    parser.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    parser.add_argument("--model", type=str, default="cpc",
                          help="Pre-trained model architecture ('cpc' [default] or 'wav2vec2').")
    parser.add_argument("--boundaryDetector", type=str, default="kreuk",
                          help="Which boundary detector to use: jch or kreuk")
    parser.add_argument('--gru_level', type=int, default=-1,
                        help='Hidden level of the LSTM autoregressive model to be taken'
                        '(default: -1, last layer).')

    args = parser.parse_args(argv)
    if args.nGPU < 0:
        args.nGPU = torch.cuda.device_count()

    args.load = [str(Path(x).resolve()) for x in args.load]
    # args.pathCheckpoint = str(Path(args.pathCheckpoint).resolve())

    return args


def main(argv):
    args = parse_args(argv)

    seqNames, speakers = findAllSeqs(args.pathDB,
                                     extension=args.file_extension,
                                     loadCache=not args.ignore_cache)

    def loadCriterion(pathCheckpoint, downsampling, nSpeakers):
        _, _, locArgs = fl.getCheckpointData(os.path.dirname(pathCheckpoint))
        criterion = getCriterion(locArgs, downsampling, nSpeakers)
        state_dict = torch.load(pathCheckpoint, 'cpu')
        criterion.load_state_dict(state_dict["cpcCriterion"])
        return criterion

    def loadCPCFeatureMaker(pathCheckpoint, gru_level=-1, get_encoded=False, keep_hidden=True):
        """
        Load CPC Feature Maker from CPC checkpoint file.
        """
        # Set LSTM level
        if gru_level is not None and gru_level > 0:
            updateConfig = argparse.Namespace(nLevelsGRU=gru_level)
        else:
            updateConfig = None
        # Load CPC model
        model, nHiddenGar, nHiddenEncoder = fl.loadModel(pathCheckpoint, updateConfig=updateConfig)
        # Keep hidden units at LSTM layers on sequential batches
        model.gAR.keepHidden = keep_hidden
        return model, nHiddenGar, nHiddenEncoder

    if args.gru_level is not None and args.gru_level > 0:
        model, hiddenGAR, hiddenEncoder = loadCPCFeatureMaker(args.load, gru_level=args.gru_level)
    else:
        model, hiddenGAR, hiddenEncoder = fl.loadModel(args.load)

    dimFeatures = hiddenEncoder if args.get_encoded else hiddenGAR

    phoneLabels, nPhones = parseSeqLabels(args.pathPhone)
    wordLabels = None
    if args.pathWords is not None:
        wordLabels, nWords = parseSeqLabels(args.pathWords)
    model.cuda()
    downsampling = model.cpc.gEncoder.DOWNSAMPLING if isinstance(model, CPCModelNullspace) else model.gEncoder.DOWNSAMPLING
    model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    criterion = loadCriterion(args.load[0], downsampling, len(speakers))
    criterion = torch.nn.DataParallel(criterion, device_ids=range(args.nGPU)).cuda()

    # Dataset
    if args.debug:
        seqNames = seqNames[:100]

    db = AudioBatchData(args.pathDB, args.size_window, seqNames,
                        phoneLabels, len(speakers), nProcessLoader=args.nProcessLoader, wordLabelsDict=wordLabels)

    batchSize = args.batchSizeGPU * args.nGPU
    dataLoader = db.getDataLoader(batchSize, 'sequential', False, numWorkers=0)

    # Checkpoint directory
    pathCheckpoint = Path(args.pathCheckpoint)
    pathCheckpoint.mkdir(exist_ok=True)
    pathCheckpoint = str(pathCheckpoint / "checkpoint")

    with open(f"{pathCheckpoint}_args.json", 'w') as file:
        json.dump(vars(args), file, indent=2)

    if args.boundaryDetector == 'jch':
        boundaryDetector = utils.jchBoundaryDetector
    elif args.boundaryDetector == 'kreuk':
        boundaryDetector = utils.kreukBoundaryDetector
    elif args.boundaryDetector == 'jhu':
        boundaryDetector = utils.jhuBoundaryDetector
    else:
        raise NotImplementedError

    run(model, criterion, boundaryDetector, dataLoader, pathCheckpoint, args.get_encoded, 
        wordSegmentation=args.pathWords is not None)



if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()

    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
