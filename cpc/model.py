# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import torch
from cpc.utils.misc import jchBoundaryDetector, jhuBoundaryDetector, kreukBoundaryDetector, compress_batch, decompress_padded_batch, getAverageSlices

###########################################
# Networks
###########################################


class IDModule(nn.Module):

    def __init__(self, *args, **kwargs):
        super(IDModule, self).__init__()

    def forward(self, x):
        return x


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(torch.Tensor(1,
                                                              numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512,
                 normMode="layerNorm",
                 linearOutput=False):

        super(CPCEncoder, self).__init__()

        validModes = ["batchNorm", "instanceNorm", "ID", "layerNorm"]
        if normMode not in validModes:
            raise ValueError(f"Norm mode must be in {validModes}")

        if normMode == "instanceNorm":
            def normLayer(x): return nn.InstanceNorm1d(x, affine=True)
        elif normMode == "ID":
            normLayer = IDModule
        elif normMode == "layerNorm":
            normLayer = ChannelNorm
        else:
            normLayer = nn.BatchNorm1d

        self.dimEncoded = sizeHidden
        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160
        self.linearOutput = linearOutput

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        if self.linearOutput:
            x = self.batchNorm4(self.conv4(x))
        else:
            x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class MFCCEncoder(nn.Module):

    def __init__(self,
                 dimEncoded):

        super(MFCCEncoder, self).__init__()
        melkwargs = {"n_mels": max(128, dimEncoded), "n_fft": 321}
        self.dimEncoded = dimEncoded
        self.MFCC = torchaudio.transforms.MFCC(n_mfcc=dimEncoded,
                                               melkwargs=melkwargs)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.MFCC(x)
        return x.permute(0, 2, 1)


class LFBEnconder(nn.Module):

    def __init__(self, dimEncoded, normalize=True):

        super(LFBEnconder, self).__init__()
        self.dimEncoded = dimEncoded
        self.conv = nn.Conv1d(1, 2 * dimEncoded,
                              400, stride=1)
        self.register_buffer('han', torch.hann_window(400).view(1, 1, 400))
        self.instancenorm = nn.InstanceNorm1d(dimEncoded, momentum=1) \
            if normalize else None

    def forward(self, x):

        N, C, L = x.size()
        x = self.conv(x)
        x = x.view(N, self.dimEncoded, 2, -1)
        x = x[:, :, 0, :]**2 + x[:, :, 1, :]**2
        x = x.view(N * self.dimEncoded, 1,  -1)
        x = torch.nn.functional.conv1d(x, self.han, bias=None,
                                       stride=160, padding=350)
        x = x.view(N, self.dimEncoded,  -1)
        x = torch.log(1 + torch.abs(x))

        # Normalization
        if self.instancenorm is not None:
            x = self.instancenorm(x)
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU,
                 segmentationThreshold,
                 smartPooling,
                 stepReduction,
                 numLevels,
                 encodeSegments,
                 minLengthSeqMinusOne,
                 mode="GRU",
                 reverse=False,
                 segmentationType='kreuk',
                 NoARonRegHead=False,
                 segmentOnContext=False):

        super(CPCAR, self).__init__()
        self.RESIDUAL_STD = 0.1

        self.heads = nn.ModuleList()

        if mode == "LSTM":
            baseNet = nn.LSTM
        elif mode == "RNN":
            baseNet = nn.RNN
        else:
            baseNet = nn.GRU
        
        for _ in range(numLevels):
                self.heads.append(baseNet(dimEncoded, dimOutput, num_layers=nLevelsGRU, batch_first=True))

        self.encodeSegments = encodeSegments
        if encodeSegments:
            self.segmentEncoder = nn.Sequential(
                nn.Linear(dimEncoded, dimEncoded),
                nn.ReLU(),
                nn.Linear(dimEncoded, dimEncoded)
            ) 

        self.hidden = [None] * numLevels
        self.keepHidden = keepHidden
        self.reverse = reverse
        self.segmentationThreshold = segmentationThreshold
        self.numLevels = numLevels
        self.smartPooling = smartPooling
        self.stepReduction = stepReduction
        self.minLengthSeqMinusOne = minLengthSeqMinusOne
        self.segmentationType = segmentationType
        self.NoARonRegHead = NoARonRegHead  
        self.segmentOnContext = segmentOnContext

    def getDimOutput(self):
        return self.heads[0].hidden_size

    def forward(self, x, label=None):
        # transformedX = []

        if not self.smartPooling and self.numLevels > 1:
            if x.size(1) % 1 / self.segmentationThreshold != 0:
                numExtraElements = x.size(1) % self.segmentationThreshold
                padValue = torch.repeat_interleave(torch.mean(x[:, -numExtraElements:, :], dim=1).view(-1, 1, x.size(2)), 
                                                    repeats=1 / self.segmentationThreshold - numExtraElements, dim=1)
                x = torch.cat((x, padValue), dim=1)

            assert x.size(1) % self.segmentationThreshold == 0

        if self.reverse:
            assert self.numLevels < 2, "reverse cpc_mode is not (yet) compatible with mACPC architecture"
            x = torch.flip(x, [1])
        try:
            for head in self.heads:
                head.flatten_parameters()
        except RuntimeError:
            pass
        
        outs = []
        hs = []

        if self.NoARonRegHead:
            o, h = x, x
        else:
            o, h = self.heads[0](x, self.hidden[0])
        
        outs.append(o)
        hs.append(h)

        for l in range(1, self.numLevels):
            if self.smartPooling:
                # minLengthSeq = max(1, int(round(2* self.minLengthSeqMinusOne * self.segmentationThreshold**l))) + 1
                minLengthSeq = self.minLengthSeqMinusOne + 1
                highLvlFeatures = o if self.segmentOnContext else x
                if self.segmentationType in ['groundTruth', 'groundTruthWError', 'groundTruthUnder', 'groundTruthOver']:
                    assert label is not None, "To use ground truth segmentation labels must be provided"
                    diffs = torch.diff(label, dim=1)
                    phoneChanges = torch.cat((torch.ones((label.shape[0], 1)).to(x.device), diffs), dim=1)
                    boundaries = torch.nonzero(phoneChanges.contiguous().view(-1), as_tuple=True)[0]
                    if self.segmentationType == 'groundTruthWError':
                        origBoundaries = boundaries[boundaries % x.size(1) != 0]
                        # noiseOffset = torch.randint_like(origBoundaries, low=-3, high=4)
                        noiseOffset = torch.randint_like(origBoundaries, low=-12, high=13)
                        newBoundaries = origBoundaries + noiseOffset
                        toFix = torch.where((origBoundaries // x.size(1) != newBoundaries // x.size(1)) | (newBoundaries < 0))[0]
                        newBoundaries[toFix] = origBoundaries[toFix]
                        boundaries = newBoundaries
                    if self.segmentationType == 'groundTruthUnder':
                        boundaries = boundaries[boundaries % x.size(1) != 0]
                        perm = torch.randperm(boundaries.size(0))
                        # idx = perm[:boundaries.size(0) // 2]
                        idx = perm[:boundaries.size(0) // 4]
                        boundaries = boundaries[idx]
                    if self.segmentationType == 'groundTruthOver':
                        for _ in range(2):
                            addedBoundaries = torch.clone(boundaries)[:-1]
                            addedBoundaries = addedBoundaries + (torch.diff(boundaries) // 2)
                            boundaries = torch.unique(torch.cat((boundaries, addedBoundaries)), sorted=True)
                    # Ensure that minibatch boundaries are preserved
                    seqEndIdx = torch.arange(0, x.size(0)*x.size(1) + 1, x.size(1), device=x.device)
                    boundaries = torch.unique(torch.cat((boundaries, seqEndIdx)), sorted=True)
                elif self.segmentationType == 'jch':
                    raise NotImplementedError
                    boundaries = jchBoundaryDetector((highLvlFeatures[:, :-1, :], 
                                                      highLvlFeatures[:, 1:, :]), 
                                                      self.segmentationThreshold**l, minLengthSeq, self.stepReduction)                                    
                elif self.segmentationType == 'kreuk':
                    boundaries = kreukBoundaryDetector((highLvlFeatures[:, :-1, :], 
                                                        highLvlFeatures[:, 1:, :]), 
                                                        self.segmentationThreshold, torch.ones(x.size(0), 
                                                        dtype=torch.int64, device=x.device) * x.size(1), minLengthSeq)
                elif self.segmentationType == 'jhu':
                    raise NotImplementedError
                    xPadded, compressedMatrices, compressedLens =  jhuBoundaryDetector(highLvlFeatures, self.segmentationThreshold, minLengthSeq)
                    packedCompressedX = torch.nn.utils.rnn.pack_padded_sequence(xPadded, compressedLens, batch_first=True, enforce_sorted=False)
                else:
                    raise NotImplementedError
                if self.segmentationType in ['jch', 'kreuk', 'groundTruth', 'groundTruthWError', 'groundTruthUnder', 'groundTruthOver']:
                    compressMatrices, compressedLens, segmentLens = getAverageSlices(boundaries, x.size(1), x.device, minLengthSeq)
                    packedCompressedX = compress_batch(
                        highLvlFeatures, compressMatrices, compressedLens, pack=True
                    )
                assert compressMatrices.shape[0] == x.shape[0]
                packedX, packedH = self.heads[l](packedCompressedX, self.hidden[l])
                statesHighLvl = decompress_padded_batch(packedX, compressMatrices)
                segments = decompress_padded_batch(packedCompressedX, compressMatrices)
                if self.encodeSegments:
                    segments = self.segmentEncoder(segments)
                outs.append({
                    'encodedData': segments,
                    'states': statesHighLvl,
                    'seqLens': compressedLens.cuda(),
                    'segmentLens': segmentLens
                })
                hs.append(packedH)
            else:
                # Random uniform pooling
                x = x.view(x.size(0), x.size(1) * self.segmentationThreshold, 1 / self.segmentationThreshold, x.size(2))
                pickedIdxs = torch.randint(x.size(2), size=(x.size(1),))
                x = x[:, torch.arange(x.size(1)), pickedIdxs, :]
                
                o, h = self.heads[l](x, self.hidden[l])
                outs.append(o)
                hs.append(h)

        if self.keepHidden:
            for l in range(self.numLevels):
                if isinstance(hs[l], tuple):
                    self.hidden[l] = tuple(x.detach() for x in hs[l])
                else:
                    self.hidden[l] = hs[l].detach()

        # For better modularity, a sequence's order should be preserved
        # by each module
        if self.reverse:
            assert self.numLevels < 2, "reverse cpc_mode is not (yet) compatible with mACPC architecture"
            outs[l] = torch.flip(outs[l], [1])
        return outs


class NoAr(nn.Module):

    def __init__(self, *args):
        super(NoAr, self).__init__()

    def forward(self, x, label):
        return [x]


class BiDIRARTangled(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRARTangled, self).__init__()
        assert(dimOutput % 2 == 0)

        self.ARNet = nn.GRU(dimEncoded, dimOutput // 2,
                            num_layers=nLevelsGRU, batch_first=True,
                            bidirectional=True)

    def getDimOutput(self):
        return self.ARNet.hidden_size * 2

    def forward(self, x):

        self.ARNet.flatten_parameters()
        xf, _ = self.ARNet(x)
        return xf


class BiDIRAR(nn.Module):
    r"""
    Research: bidirectionnal model for BERT training.
    """
    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 nLevelsGRU):

        super(BiDIRAR, self).__init__()
        assert(dimOutput % 2 == 0)

        self.netForward = nn.GRU(dimEncoded, dimOutput // 2,
                                 num_layers=nLevelsGRU, batch_first=True)
        self.netBackward = nn.GRU(dimEncoded, dimOutput // 2,
                                  num_layers=nLevelsGRU, batch_first=True)

    def getDimOutput(self):
        return self.netForward.hidden_size * 2

    def forward(self, x):

        self.netForward.flatten_parameters()
        self.netBackward.flatten_parameters()
        xf, _ = self.netForward(x)
        xb, _ = self.netBackward(torch.flip(x, [1]))
        return torch.cat([xf, torch.flip(xb, [1])], dim=2)


###########################################
# Model
###########################################


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData, label)
        return cFeature, encodedData, label

class CPCModelNullspace(nn.Module):

    def __init__(self,
                 cpc,
                 nullspace):

        super(CPCModelNullspace, self).__init__()
        self.cpc = cpc
        self.nullspace = nn.Linear(nullspace.shape[0], nullspace.shape[1], bias=False)
        self.nullspace.weight = nn.Parameter(nullspace.T)
        self.gEncoder = self.cpc.gEncoder


    def forward(self, batchData, label):
        cFeature, encodedData, label = self.cpc(batchData, label)
        cFeature = self.nullspace(cFeature)
        encodedData = self.nullspace(encodedData)
        return cFeature, encodedData, label


class ConcatenatedModel(nn.Module):

    def __init__(self, model_list):

        super(ConcatenatedModel, self).__init__()
        self.models = torch.nn.ModuleList(model_list)

    def forward(self, batchData, label):

        outFeatures = []
        outEncoded = []
        for model in self.models:
            cFeature, encodedData, label = model(batchData, label)
            outFeatures.append(cFeature)
            outEncoded.append(encodedData)
        return torch.cat(outFeatures, dim=2), \
            torch.cat(outEncoded, dim=2), label
