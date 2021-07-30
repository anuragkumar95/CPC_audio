

import torch.nn as nn
import torch
import numpy as np
import math
from collections import deque
import random

from cpc.model import seDistancesToCentroids

class CentroidModule(nn.Module):

    def __init__(self, settings):  
        super().__init__()
        self.dsLen = 0
        self.dsCountEpoch = None
        self.nextExampleNum = 0
        self.seenNr = 0
        self.chosenExamples = None
        self.kmeansInitBatches = None  # will be set below if value given
        self.kmeansReinitEachN = None  # same as above
        self.centerNorm = None # same
        self.batchUpdateQ = None  # same
        self.protoCounts = None
        self.protoSums = None
        self.debug = settings["debug"]
        self.addedChosenBatchInputs = set()
        self.chosenBatchInputs = []
        self.chosenKMeansBatches = []
        self.numCentroids = settings["numCentroids"]
        self.reprDim = settings["reprDim"]
        self.numPhones = settings["numPhones"]  # used for calculating metrics
        self.mode = settings["mode"]
        assert self.mode in ("reprInit", "beginInit", "onlineKmeans")
        if self.mode == "reprInit":
            self.initAfterEpoch = settings["initAfterEpoch"]  # initAfterEpoch can be set to -1 for no repr init, just regular
                                                              # otherwise this epoch has to be > 0 (as first epoch is DS counting)
            print(f"CENTROID INIT AFTER EPOCH: {self.initAfterEpoch}")
        if self.mode in ("reprInit", "beginInit"):
            # needed also in reprInit case so that params can be added to optimizer
            # [!!] this is initialized here, but only trained in cpcModel forward (because of the loss)!
            self.protos = nn.Parameter(torch.randn((self.numCentroids, self.reprDim), requires_grad=True).cuda() / (5. * math.sqrt(self.reprDim)), requires_grad=True)
        if self.mode == "onlineKmeans":  # modification of what is called "sequential k-means" in literature (with fixed memory of some batches added)
            self.initAfterEpoch = settings["initAfterEpoch"]  # this epoch has to be > 0 (as first epoch is DS counting)
            self.firstInitNoIters = settings["firstInitNoIters"]
            self.kmeansInitIters = settings["kmeansInitIters"]
            self.kmeansInitBatches = settings["kmeansInitBatches"]
            if self.kmeansInitIters or self.kmeansInitBatches:
                assert self.kmeansInitIters and self.kmeansInitBatches
            self.kmeansReinitEachN = settings["kmeansReinitEachN"]
            self.kmeansReinitUpTo = settings["kmeansReinitUpTo"]
            print(f"CENTROID INIT AFTER EPOCH: {self.initAfterEpoch}")
            self.keepBatches = settings["onlineKmeansBatches"]
            self.keepBatchesLongTerm = settings["onlineKmeansBatchesLongTerm"]
            self.keepBatchesLongTermWeight = settings["onlineKmeansBatchesLongTermWeight"]
            self.centerNorm = settings["centerNorm"]
            self.batchRecompute = settings["batchRecompute"]
            if self.batchRecompute:
                self.batchUpdateQ = deque()
            self.protos = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
            # [!] to resume, memory batches (and also sums etc.) would need to be saved in state - there are much of them, this is not done
            #     only centroids are saved
            self.currentGlobalBatch = 0
            self.protoCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
            self.protoSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
            self.lastKmBatches = {}  # format:     batch_num: (sums_of_points_assigned_to_each_center, num_points_assigned_to_each_center, batch audio)
            self.longTermKmBatches = {}
            
    # before encodings
    def inputsBatchUpdate(self, batch, epochNrs, cpcModel):
        
        if self.debug:
            print(f"--> INPUT BATCH UPDATE epNrs. {epochNrs}, {batch.shape}")
        
        with torch.no_grad():
            self.last_input_batch = batch.clone().detach()
            # can be several shorter batches, one per speaker or so, but it is like
            # inputsBatchUpdate, encodingsBatchUpdate are in turns, always
        

    def encodingsBatchUpdate(self, batch, epochNrs, cpcModel, label=None):
        epoch, totalEpochs = epochNrs
        batch = batch.clone().detach()
        if self.dsCountEpoch is None or self.dsCountEpoch == epoch:
            self.dsCountEpoch = epoch
            if self.dsLen == 0:
                print(f"--> COUNTING DS LEN during epoch {epoch}")
            self.dsLen += batch.shape[0] * batch.shape[1]

        if self.mode in ("reprInit", "onlineKmeans") and epoch == self.initAfterEpoch:  # this epoch has to be > 0 (as first epoch is DS counting)
            if self.chosenExamples is None:
                if self.kmeansInitBatches:
                    randNr = max(self.kmeansInitBatches, self.numCentroids)
                else:
                    randNr = self.numCentroids
                self.chosenExamples = np.random.choice(self.dsLen, randNr*2)  # choosing too much in case of repetitions, will cut below
                self.chosenExamples = list(set(self.chosenExamples))
                random.shuffle(self.chosenExamples)
                self.chosenExamples = self.chosenExamples[:randNr]  # cutting too much choises - fixing out possible repetitions
                if self.kmeansInitBatches:
                    self.chosenKMeansCandidateNrs = set(self.chosenExamples[:self.kmeansInitBatches])
                    print(f"--> CHOOSING {self.kmeansInitBatches} BATCHES FOR K-MEANS INIT POINTS;"
                           " will make k-means init with EXAMPLES as starting centers")
                else:
                    self.chosenKMeansCandidateNrs = None
                self.chosenCentroidsCandidateNrs = set(self.chosenExamples[:self.numCentroids])
                self.chosenExamples = sorted(list(self.chosenExamples))
                self.chosenExamples.append(1000000000000000000000000)  # for convenience and less cases below
                print(f"--> CHOOSING {self.numCentroids} EXAMPLES FOR CENTROID INIT, DSLEN {self.dsLen}: {sorted(list(self.chosenCentroidsCandidateNrs))}")
            numHere = batch.shape[0] * batch.shape[1]  # not assuming each batch has same size because it does not sometimes
            candidateNr = self.chosenExamples[self.nextExampleNum]
            addedThisBatch = False
            while candidateNr < self.seenNr + numHere:
                offset = candidateNr - self.seenNr
                lineNr = offset // batch.shape[1]
                lineOffset = offset % batch.shape[1]
                if self.debug:
                    print(f"-> adding candidate / its batch, candidateNr {candidateNr}, batch data end {self.seenNr + numHere}, batch data shape {self.last_input_batch.shape}")
                with torch.no_grad():  # self.last_input_batch already detached
                    if candidateNr in self.chosenCentroidsCandidateNrs and candidateNr not in self.addedChosenBatchInputs:
                        self.addedChosenBatchInputs.add(candidateNr)
                        self.chosenBatchInputs.append((self.last_input_batch.clone().cpu(), lineNr, lineOffset))
                        print(f"--> ADDED BATCH EXAMPLE #{self.nextExampleNum}")
                    if self.chosenKMeansCandidateNrs and candidateNr in self.chosenKMeansCandidateNrs and not addedThisBatch:  # both this and above can happen
                        self.chosenKMeansBatches.append(self.last_input_batch.clone().cpu())
                        addedThisBatch = True
                    
                self.nextExampleNum += 1
                candidateNr = self.chosenExamples[self.nextExampleNum]
            self.seenNr += numHere  #batch.shape[0] * batch.shape[1]

        if self.mode == "onlineKmeans" and epoch > self.initAfterEpoch:

            if self.debug:
                print(f"BATCH UPDATE onlineKmeans, memory keys: {self.lastKmBatches.keys()}, long term memory keys: {self.longTermKmBatches.keys()}")

            if self.centerNorm:
                batch = self.normLen(batch) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)

            distsSq = seDistancesToCentroids(batch, self.protos, debug=self.debug)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)

            # add new batch data
            batchSums, closestCounts, labelCounts = self.getBatchSums(batch, closest, label=label)
            self.protoSums += batchSums
            self.protoCounts += closestCounts
            batchToRemember = self.last_input_batch.clone().cpu() if self.batchRecompute else None
            self.lastKmBatches[self.currentGlobalBatch] = (batchSums.cpu(), closestCounts.cpu(), batchToRemember)  # on batchToRemember .cpu() above if not None
            if self.batchRecompute:
                self.batchUpdateQ.append(self.currentGlobalBatch)
            if self.keepBatchesLongTerm:
                weightedSums = self.keepBatchesLongTermWeight*batchSums
                weightedCounts = self.keepBatchesLongTermWeight*closestCounts
                self.protoSums += weightedSums
                self.protoCounts += weightedCounts
                self.longTermKmBatches[self.currentGlobalBatch] = (weightedSums.cpu(), weightedCounts.cpu())

            # subtract old out-of-the-window batch data
            oldBatch = self.currentGlobalBatch - self.keepBatches
            if oldBatch in self.lastKmBatches:
                oldBatchSums, oldBatchCounts, _ = self.lastKmBatches[oldBatch]
                oldBatchSums = oldBatchSums.cuda()
                oldBatchCounts = oldBatchCounts.cuda()
                self.protoSums -= oldBatchSums
                self.protoCounts -= oldBatchCounts
                del self.lastKmBatches[oldBatch]
            if self.keepBatchesLongTerm:
                oldBatch = self.currentGlobalBatch - self.keepBatchesLongTerm
                if oldBatch in self.longTermKmBatches:
                    oldBatchSums, oldBatchCounts = self.longTermKmBatches[oldBatch]
                    oldBatchSums = oldBatchSums.cuda()
                    oldBatchCounts = oldBatchCounts.cuda()
                    self.protoSums -= oldBatchSums
                    self.protoCounts -= oldBatchCounts
                    del self.longTermKmBatches[oldBatch]

            if self.batchRecompute:
                self.updateBatches(epochNrs, cpcModel)

            # re-average centroids
            with torch.no_grad():  # just in case it tries to compute grad
                if self.currentGlobalBatch >= self.keepBatches:
                    self.protos = self.protoSums / torch.clamp(self.protoCounts.view(-1,1), min=1)
                if self.centerNorm:
                    with torch.no_grad():
                        self.protos = self.normLen(self.protos)

            self.currentGlobalBatch += 1

            return {"labelCounts": labelCounts} if self.currentGlobalBatch >= self.keepBatches else None

    def normLen(self, tens):
        # normalization, but not if very very short - to prevent problems during training
        tensLens = torch.sqrt(torch.clamp((tens*tens).sum(-1), min=0))
        return tens / torch.clamp(tensLens.view(*(tensLens.shape), 1), min=1)

    def updateBatches(self, epochNrs, cpcModel):
        updated = 0
        while len(self.batchUpdateQ) > 0 and updated < self.batchRecompute:
            batchNr = self.batchUpdateQ.popleft()
            if batchNr not in self.lastKmBatches:
                continue  # old batch out of window, no update
            oldBatchSums, oldBatchCounts, batch = self.lastKmBatches[batchNr]
            oldBatchSums = oldBatchSums.cuda()
            oldBatchCounts = oldBatchCounts.cuda()
            batch = batch.cuda()
            with torch.no_grad():
                encoded_data = cpcModel(batch, None, None, None, None, epochNrs, False, True)
            if self.centerNorm:
                encoded_data = self.normLen(encoded_data) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            distsSq = seDistancesToCentroids(encoded_data, self.protos, debug=self.debug)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)
            batchSums, closestCounts, _ = self.getBatchSums(encoded_data, closest)
            self.protoSums -= oldBatchSums
            self.protoCounts -= oldBatchCounts
            self.protoSums += batchSums
            self.protoCounts += closestCounts
            self.lastKmBatches[batchNr] = (batchSums.cpu(), closestCounts.cpu(), batch.cpu())
            self.batchUpdateQ.append(batchNr)
            if self.centerNorm:
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            if self.debug:
                print("UPDATED for batch nr:", batchNr)
            updated += 1

            
            
    def getBatchSums(self, batch, closest, label=None):
        # batch B x n x dim
        # closest B x n

        # commented out code below was for version without iterating over clusters, 
        # but it needed much bigger matrices (because a[closest] += representations doesn;t work in pytorch,
        # //closest is of dim B x N (indication of closest centroid to each repr), repr B x N x Dim
        # torch only adds LAST representation with each closest centroid = i to a[i] and NOT all for some reason,
        # which I consider utter nonsense, and to avoid this and do all at once you need to add one extra big dimension,
        # and AFAIK torch doesn't provide and non-coding-in-CUDA reasonable way to do this, I'd say, natural simple summing operation)
        # and therefore this commented out option was slower after all 
        ##batchExtended = torch.zeros(batch.shape[0], batch.shape[1], self.protos.shape[0], batch.shape[2], dtype=torch.float32).cuda()
        ##firstDim = torch.arange(batch.shape[0]).repeat_interleave(batch.shape[1]).view(batch.shape[0], batch.shape[1])
        ##print(firstDim)
        ##secondDim = torch.arange(batch.shape[1]).repeat(batch.shape[0]).view(batch.shape[0], batch.shape[1])
        ##print(batchExtended.dtype, batch.dtype)
        ##batchExtended[firstDim, secondDim, closest, :] = batch
        ##batchSums = batchExtended.sum(dim=(0,1))  #[closest] += batch  # only takes last value for index, pathetic
        batchSums = torch.zeros(self.protos.shape[0], batch.shape[2], dtype=torch.float32).cuda()
        for i in range(self.protos.shape[0]):
            batchSums[i] += batch[closest==i, :].sum(dim=(0))
        indices, indicesCounts = torch.unique(closest, return_counts=True)
        closestCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
        closestCounts[indices] += indicesCounts
        if label is not None and self.numPhones:
            label = label.cuda()
            ##labelsAssignment = torch.zeros(batch.shape[0], batch.shape[1], self.protos.shape[0], self.numPhones, dtype=torch.float32).cuda()
            labelsSums = torch.zeros(self.protos.shape[0], self.numPhones, dtype=torch.float32).cuda()
            ##labelsAssignment[firstDim, secondDim, closest, label[firstDim,secondDim]] += 1
            for i in range(self.protos.shape[0]):
                labelclosest = label[closest==i]
                lindices, lindicesCounts = torch.unique(labelclosest, return_counts=True)
                lclosestCounts = torch.zeros(self.numPhones, dtype=torch.float32).cuda()
                lclosestCounts[lindices] += lindicesCounts
                labelsSums[i] += lclosestCounts
            ##labelsSums = labelsAssignment.sum(dim=(0,1))
            return batchSums, closestCounts, labelsSums
        
        return batchSums, closestCounts, None

        

    def printLens(self):
        with torch.no_grad():
            print((self.protos*self.protos).sum(dim=-1))
            
    def getDM(self, epoch):
        protosHere = self.centersForStuff(epoch)
        if protosHere is None:
            return None
        DMsq = seDistancesToCentroids(protosHere, protosHere, debug=self.debug).view(protosHere.shape[0], protosHere.shape[0])
        return torch.sqrt(torch.clamp(DMsq, min=0))

    def epochUpdate(self, epochNrs, cpcModel):  # after that epoch
        epoch, allEpochs = epochNrs
        if self.mode in ("reprInit", "onlineKmeans"):
            if epoch == self.initAfterEpoch or \
                (epoch > self.initAfterEpoch and self.kmeansReinitEachN and (epoch - self.initAfterEpoch) % self.kmeansReinitEachN == 0 and \
                (not self.kmeansReinitUpTo or epoch < self.kmeansReinitUpTo)):   

                with torch.no_grad():

                    self.currentGlobalBatch = 0  # to prevent pushing with incomplete means
                    # to remove info that will be invalid with new clusters
                    self.lastKmBatches = {}
                    self.longTermKmBatches = {}
                    self.protoCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
                    self.protoSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
                    if self.batchUpdateQ is not None:
                        self.batchUpdateQ.clear()

                    print("K-MEANS CENTERS INIT/REINIT FROM REPRESENTATIONS")
                    self.initKmeansCenters(epochNrs, cpcModel)  # initialize centroids
                    if self.kmeansInitBatches and (not self.firstInitNoIters or epoch != self.initAfterEpoch):
                        print("K-MEANS CENTERS INIT/REINIT K-MEANS IMPROVE BY k-means ITERATIONS")
                        for i in range(self.kmeansInitIters):  # perform k-means with initizalized centroids
                            print("new kmeans epoch")
                            self.kmeansEpoch(epochNrs, cpcModel)  # performs one epoch, moving the centroids

            
    def initKmeansCenters(self, epochNrs, cpcModel):
        for i, (batchData, lineNr, lineOffset) in enumerate(self.chosenBatchInputs):
            if self.debug:
                print(f"running k means init on a batch; candidate representation inside will be in: line {lineNr}, lineOffset {lineOffset}; batch data shape: {batchData.shape}")
            with torch.no_grad():
                batchData = batchData.cuda()
                # last arg tells to only run encoding part
                encoded_data = cpcModel(batchData, None, None, None, None, epochNrs, False, True)  # c_feature, encoded_data, label, pushLoss
                if self.debug:
                    if encoded_data is None:
                        print("encoded_data None!")
                    else:
                        print(f"encoded_data representations shape: {encoded_data.shape}")
                self.protos[i] = encoded_data[lineNr,lineOffset]
                # [!!!] here it's not normed, it's normed before any distance operation or before giving it outside
                print(f"--> CENTROID INIT EXAMPLE #{i}: sqlen {(self.protos[i]*self.protos[i]).sum(-1)}")  #"; {self.protos[i]}")


    def kmeansEpoch(self, epochNrs, cpcModel):
        # this assumes centroids are already initialized
        newCentersSums = torch.zeros((self.numCentroids, self.reprDim), dtype=torch.float32).cuda()
        newCentersCounts = torch.zeros(self.protos.shape[0], dtype=torch.float32).cuda()
        print(f"ACTUAL BATCHES for k-means init epoch: {len(self.chosenKMeansBatches)}")
        for i, batch in enumerate(self.chosenKMeansBatches):
            batch = batch.cuda()
            encoded_data = cpcModel(batch, None, None, None, None, epochNrs, False, True)
            if self.centerNorm:
                encoded_data = self.normLen(encoded_data) 
                with torch.no_grad():
                    self.protos = self.normLen(self.protos)
            distsSq = seDistancesToCentroids(encoded_data, self.protos, debug=self.debug)
            distsSq = torch.clamp(distsSq, min=0)
            #dists = torch.sqrt(distsSq)
            closest = distsSq.argmin(-1)
            # add new batch data
            batchSums, closestCounts, _ = self.getBatchSums(encoded_data, closest)
            newCentersSums += batchSums
            newCentersCounts += closestCounts
        with torch.no_grad():  # just in case it tries to compute grad
            self.protos = newCentersSums / torch.clamp(newCentersCounts.view(-1,1), min=1)

    def centersForSave(self):
        return self.protos.clone().detach().cpu()

    def centersForStuff(self, epoch):
        # if None returned, things will not do stuff like pushing to centers; otherwise will
        if self.centerNorm:
            with torch.no_grad():
                self.protos = self.normLen(self.protos)
        if self.mode == "reprInit":
            if epoch <= self.initAfterEpoch:
                return None
            else:
                return self.protos
        if self.mode == "beginInit":
            return self.protos

        if self.mode == "onlineKmeans":
            # count starts after init, but more fireproof with epoch check
            if self.debug:
                print(f"onlineKmeans centersForStuff request; initAfterEpoch {self.initAfterEpoch}, currentGlobalBatch {self.currentGlobalBatch}")
            if epoch > self.initAfterEpoch and self.currentGlobalBatch >= self.keepBatches:
                if self.debug:
                    print("centroids returned")
                return self.protos
            else:
                if self.debug:
                    print("None returned")
                return None
        






if __name__ == "__main__":

    # online kmeans test

    # can happen that only points from 1 batch are chosen for init and this can make it needed to rerun (randomization)
    # but should be rare
    
    batch = torch.tensor([[[7,7], [2,2], [3,3]], [[7,7], [2,2], [3,3]]], dtype=float)
    
    cm = CentroidModule({
        "mode": "onlineKmeans",
        "onlineKmeansBatches": 2,  
        "reprDim": 2,
        "numCentroids": 4,
        "initAfterEpoch": 1,
        "debug": True,
        "numPhones": 10,
        "firstInitNoIters": False,
        "kmeansInitIters": 3,
        "kmeansInitBatches": 5,  # can happen that only 1 will be drawn (all 3 from same batch) and then only 3 centroids will be there 
        "kmeansReinitEachN": None,
        "kmeansReinitUpTo": None,
        "onlineKmeansBatchesLongTerm": None,
        "onlineKmeansBatchesLongTermWeight": None,
        "centerNorm": False,
        "pointNorm": False,
        "batchRecompute": False})


    # separate check
    distsSq = torch.tensor([[[1,4], [4,1], [4,1]], [[1,4], [4,1], [4,1]]], dtype=float)  # only care for argmin
    closest = distsSq.argmin(-1)
    print(cm.getBatchSums(batch.cuda(), closest.cuda()))  


    # sometimes needs to be retried to be valuable as things are randomized
    # the idea is so that there are 3 "main" centroids, around (2,2), (18,18/0 and (32,32),
    # 4th centroid can easily zero out (no closest points at some moment) in this setting quickly

    batch1 = torch.tensor([[[17,17], [2,2], [31,31]], [[17,17], [2,2], [31,31]]], dtype=float).cuda()
    batch2 = torch.tensor([[[18,18], [2,2], [32,32]], [[18,18], [2,2], [32,32]]], dtype=float).cuda()
    batch3 = torch.tensor([[[19,19], [2,2], [33,33]], [[19,19], [2,2], [33,33]]], dtype=float).cuda()

    # onlyConv is just-conv-encode forward variant 
    cpcModelFake = lambda batch, a, b, c, d, e, f, onlyConv: batch if onlyConv else (None, None, batch, None, None, None, None, None, None, None)

    for ep in range(4):
        eps = (ep,4)
        for batch in (batch1, batch2, batch3):
            cm.inputsBatchUpdate(batch, eps, cpcModelFake)
            cm.encodingsBatchUpdate(batch, eps, cpcModelFake)
            print("\n---> CENTERS AFTER BATCH UPDATE: ", cm.centersForStuff(ep))
        cm.epochUpdate(eps, cpcModelFake)
        print("\n===> CENTERS AFTER EPOCH: ", cm.centersForStuff(ep))


    print("--------------------------")

    batch = torch.tensor([[[7,8], [7,6], [0,3]], [[1,7], [2,2], [0,1]]], dtype=float)

    cm = CentroidModule({
        "mode": "onlineKmeans",
        "onlineKmeansBatches": 2,  
        "reprDim": 2,
        "numCentroids": 4,
        "initAfterEpoch": 1,
        "debug": True,
        "numPhones": 10,
        "firstInitNoIters": False,
        "kmeansInitIters": 3,
        "kmeansInitBatches": 5,  # can happen that only 1 will be drawn (all 3 from same batch) and then only 3 centroids will be there 
        "kmeansReinitEachN": None,
        "kmeansReinitUpTo": None,
        "onlineKmeansBatchesLongTerm": None,
        "onlineKmeansBatchesLongTermWeight": None,
        "centerNorm": True,
        "pointNorm": True,
        "batchRecompute": False})


    # separate check
    distsSq = torch.tensor([[[1,4], [1,4], [4,1]], [[4,1], [1,4], [4,1]]], dtype=float)  # only care for argmin
    closest = distsSq.argmin(-1)
    print(cm.getBatchSums(batch.cuda(), closest.cuda()))


    # sometimes needs to be retried to be valuable as things are randomized
    # the idea is so that there are 3 "main" centroids, around: (0.71, 0.71), (0,1) and (-1,0),
    # 4th centroid can easily zero out (no closest points at some moment) in this setting quickly

    batch1 = torch.tensor([[[-21,1], [0,20], [31,31]], [[0,20], [-50,2], [31,31]]], dtype=float).cuda()
    batch2 = torch.tensor([[[18,18], [1,21], [-100,5]], [[1,21], [-1234,41], [32,32]]], dtype=float).cuda()
    batch3 = torch.tensor([[[19,19], [-55,4], [33,33]], [[-7,0], [2,42], [33,33]]], dtype=float).cuda()

    # onlyConv is just-conv-encode forward variant 
    cpcModelFake = lambda batch, a, b, c, d, e, f, onlyConv: batch if onlyConv else (None, None, batch, None, None, None, None, None, None, None)

    for ep in range(4):
        eps = (ep,4)
        for batch in (batch1, batch2, batch3):
            cm.inputsBatchUpdate(batch, eps, cpcModelFake)
            cm.encodingsBatchUpdate(batch, eps, cpcModelFake)
            print("\n---> CENTERS AFTER BATCH UPDATE: ", cm.centersForStuff(ep))
        cm.epochUpdate(eps, cpcModelFake)
        print("\n===> CENTERS AFTER EPOCH: ", cm.centersForStuff(ep))    










