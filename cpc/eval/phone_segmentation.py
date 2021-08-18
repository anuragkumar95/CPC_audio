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
import cpc.feature_loader as fl
import cpc.utils.misc as utils
from cpc.dataset import AudioBatchData, findAllSeqs, filterSeqs, parseSeqLabels
from cpc.model import CPCModelNullspace
# import pandas as pd

def evalPhoneSegmentation(featureMaker, boundaryDetector, dataLoader, labelKey="speaker", onEncodings=True, toleranceInFrames=2):

    featureMaker.eval()
    logs = {"precision": 0, "recall": 0, "f1": 0, "r": 0}
    EPS = 1e-7
    segmentationParamRange = np.arange(0.01, 0.15, 0.01)

    # results = []
    for step, fulldata in tqdm.tqdm(enumerate(dataLoader)):
        with torch.no_grad():
            batch_data, labelData = fulldata
            label = labelData[labelKey]
            cFeature, encodedData, _ = featureMaker(batch_data, None)
            cFeature = cFeature[0]
        maxRval = 0

        diffs = torch.diff(label, dim=1)
        phone_changes = torch.cat((torch.ones((label.shape[0], 1), device=label.device), diffs), dim=1)
        trueBoundaries = torch.nonzero(phone_changes.contiguous().view(-1), as_tuple=True)[0]
        # label = torch.cat([label[:, 0].view(-1, 1), label], dim=1)
        # trueBoundaries = torch.where(torch.diff(label.view(-1)) != 0)[0]
        # Ensure that minibatch boundaries are preserved
        seqEndIdx = torch.arange(0, encodedData.size(0)*encodedData.size(1) + 1, encodedData.size(1))
        trueBoundaries = torch.unique(torch.cat((trueBoundaries, seqEndIdx)), sorted=True)

        for segmentationParam in segmentationParamRange:
            if onEncodings:
                predictedBoundaries = boundaryDetector(encodedData, segmentationParam, justSegmenter=True).cpu()
            else:
                predictedBoundaries = boundaryDetector(cFeature, segmentationParam, justSegmenter=True).cpu()
            
            precisionCounter = 0
            recallCounter = 0

            for predictedBoundary in predictedBoundaries:
                minDist = torch.min(torch.abs(trueBoundaries - predictedBoundary))
                precisionCounter += (minDist <= toleranceInFrames)

            for trueBoundary in trueBoundaries:
                minDist = torch.min(torch.abs(predictedBoundaries - trueBoundary))
                detected = (minDist <= toleranceInFrames)
                # if not detected:
                #     results.append({
                #         'prev': label.view(-1)[trueBoundary - 1],
                #         'post': label.view(-1)[trueBoundary] 
                #     })
                recallCounter += detected

            precision = precisionCounter / (len(predictedBoundaries) + EPS)
            recall = recallCounter / (len(trueBoundaries) + EPS)
            f1 = 2 * (precision * recall) / (precision + recall + EPS)
            os = recall / (precision + EPS) - 1
            r1 = np.sqrt((1 - recall) ** 2 + os ** 2)
            r2 = (-os + recall - 1) / (np.sqrt(2))
            rVal = 1 - (np.abs(r1) + np.abs(r2)) / 2
            
            if rVal > maxRval:
                maxRval = rVal
                bestPrecision = precision
                bestRecall = recall
                bestF1 = f1
        logs["precision"] += bestPrecision.view(1).numpy()
        logs["recall"] += bestRecall.view(1).numpy()
        logs["f1"] += bestF1.view(1).numpy()
        logs["r"] += maxRval.view(1).numpy()
    # results = pd.DataFrame(results)
    # results.to_csv('not_detected.csv')
    logs = utils.update_logs(logs, step)

    return logs


def run(featureMaker,
        boundaryDetector,
        trainLoader,
        valLoader,
        pathCheckpoint,
        onEncodings,
        labelKey="speaker"):
    # print("Training dataset %d batches, Validation dataset %d batches" % (len(trainLoader), len(valLoader)))
    print("Validation dataset %d batches" % len(valLoader))
    # logsTrain = evalPhoneSegmentation(featureMaker, boundaryDetector, trainLoader, labelKey, onEncodings)
    logsVal = evalPhoneSegmentation(featureMaker, boundaryDetector, valLoader, labelKey, onEncodings)
    # utils.show_logs("Training stats", logsTrain)
    utils.show_logs("Validation stats", logsVal)
    # for key, value in dict(logsTrain).items():
        # if isinstance(value, np.ndarray):
            # value = value.tolist()
        # logsTrain[key] = value
    for key, value in dict(logsVal).items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        logsVal[key] = value
    # utils.save_logs(logsTrain, f"{pathCheckpoint}_logsTrain.json")
    utils.save_logs(logsVal, f"{pathCheckpoint}_logsVal.json")


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Linear separability trainer'
                                     ' (default test in speaker separability)')
    parser.add_argument('pathDB', type=str,
                        help="Path to the directory containing the audio data.")
    parser.add_argument('pathTrain', type=str,
                        help="Path to the list of the training sequences.")
    parser.add_argument('pathVal', type=str,
                        help="Path to the list of the test sequences.")
    parser.add_argument('load', type=str, nargs='*',
                        help="Path to the checkpoint to evaluate.")
    parser.add_argument('--pathPhone', type=str, default=None,
                        help="Path to the phone labels. If given, will"
                        " compute the phone separability.")
    # parser.add_argument('--pathCheckpoint', type=str, default='out',
    #                     help="Path of the output directory where the "
    #                     " checkpoints should be dumped.")
    parser.add_argument('--nGPU', type=int, default=-1,
                        help='Bumber of GPU. Default=-1, use all available '
                        'GPUs')
    parser.add_argument('--batchSizeGPU', type=int, default=8,
                        help='Batch size per GPU.')
    parser.add_argument('--debug', action='store_true',
                        help='If activated, will load only a small number '
                        'of audio data.')
    parser.add_argument('--file_extension', type=str, default=".flac",
                        help="Extension of the audio files in pathDB.")
    parser.add_argument('--get_encoded', action='store_true',
                        help="If activated, will work with the output of the "
                        " convolutional encoder (see CPC's architecture).")
    parser.add_argument('--ignore_cache', action='store_true',
                        help="Activate if the sequences in pathDB have"
                        " changed.")
    parser.add_argument('--size_window', type=int, default=20480,
                        help="Number of frames to consider in each batch.")
    parser.add_argument('--n_process_loader', type=int, default=8,
                          help='Number of processes to call to load the '
                          'dataset')
    parser.add_argument('--max_size_loaded', type=int, default=4000000000,
                          help='Maximal amount of data (in byte) a dataset '
                          'can hold in memory at any given time')
    parser.add_argument("--model", type=str, default="cpc",
                          help="Pre-trained model architecture ('cpc' [default] or 'wav2vec2').")
    parser.add_argument("--boundaryDetector", type=str, default="jch",
                          help="Which boundary detector to use: jch, kreuk, or jhu")
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

        # Build CPC Feature Maker from CPC model
        #featureMaker = fl.FeatureModule(model, get_encoded=get_encoded)

        #return featureMaker
        return model, nHiddenGar, nHiddenEncoder

    if args.gru_level is not None and args.gru_level > 0:
        model, hidden_gar, hidden_encoder = loadCPCFeatureMaker(args.load, gru_level=args.gru_level)
    else:
        model, hidden_gar, hidden_encoder = fl.loadModel(args.load)

    dim_features = hidden_encoder if args.get_encoded else hidden_gar

    phone_labels, n_phones = parseSeqLabels(args.pathPhone)
    label_key = 'phone'
    
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(args.nGPU))

    # Dataset
    seq_train = filterSeqs(args.pathTrain, seqNames)
    seq_val = filterSeqs(args.pathVal, seqNames)

    if args.debug:
        seq_train = seq_train[:1000]
        seq_val = seq_val[:100]

    # db_train = AudioBatchData(args.pathDB, args.size_window, seq_train,
                            #   phone_labels, len(speakers), nProcessLoader=args.n_process_loader,
                                #   MAX_SIZE_LOADED=args.max_size_loaded)
    db_val = AudioBatchData(args.pathDB, args.size_window, seq_val,
                            phone_labels, len(speakers), nProcessLoader=args.n_process_loader)

    batch_size = args.batchSizeGPU * args.nGPU

    train_loader = None
    # train_loader = db_train.getDataLoader(batch_size, "uniform", True,
                                        #   numWorkers=0)

    val_loader = db_val.getDataLoader(batch_size, 'sequential', False,
                                      numWorkers=0)

    # Checkpoint directory
    checkpoint_dir = os.path.dirname(args.load[0])
    checkpoint_no = args.load[0].split('_')[-1][:-3]
    pathCheckpoint = f"{checkpoint_dir}/phoneSeg{args.boundaryDetector}_{checkpoint_no}"
    if args.get_encoded:
        pathCheckpoint += '_onEnc'
    pathCheckpoint = Path(pathCheckpoint)
    pathCheckpoint.mkdir(exist_ok=True)
    pathCheckpoint = str(pathCheckpoint / "checkpoint2")

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

    run(model, boundaryDetector, train_loader, val_loader,
        pathCheckpoint, args.get_encoded, labelKey=label_key)



if __name__ == "__main__":
    #import ptvsd
    #ptvsd.enable_attach(('0.0.0.0', 7310))
    #print("Attach debugger now")
    #ptvsd.wait_for_attach()

    torch.multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    main(args)
