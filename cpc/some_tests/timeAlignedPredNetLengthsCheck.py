

import torch
from cpc.criterion.soft_align import TimeAlignedPredictionNetwork



#### this check is intended for checking if predicted lengths (moreLengths) are calcualted correctly
#### for weighting predictions there is another check with smaller data in a separate file

### predictedLengths used in this file are:
### raw: [[-0.6,-0.6,-0.6,-0.6,-0.6,-0.6,0.4], [-0.6,-0.6,-0.4,-0.2,0.,-0.6,-0.6]]
### after [-1,1] (LSTM) -> [0,1] mapping: [[0.2,0.2,0.2,0.2,0.2,0.2,0.7], [0.2,0.2,0.3,0.4,0.5,0.2,0.2]]


print("---> 1")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", debug=True).cuda()

# 2nd dim are predLengths - like with option to pass detached lengths to predictors
featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors
#candidates = torch.zeros_like(featc).view(1,featc.shape[0],1,featc.shape[1],featc.shape[2]).repeat(3,1,5,1,1).cuda()

print(featc.shape)  #, candidates.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)  #candidates[:,:,:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 2")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", debug=True,
    teachOnlyLastFrameLength=True).cuda()

featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 3")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", debug=True,
    teachLongPredsUniformlyLess=True).cuda()

featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 4")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", debug=True,
    teachLongPredsSqrtLess=True).cuda()

featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 5")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", 
                                    teachLongPredsSqrtLess=True, lengthsGradReweight=5., debug=True).cuda()

# 2nd dim are predLengths - like with option to pass detached lengths to predictors
featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors
#candidates = torch.zeros_like(featc).view(1,featc.shape[0],1,featc.shape[1],featc.shape[2]).repeat(3,1,5,1,1).cuda()

print(featc.shape)  #, candidates.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)  #candidates[:,:,:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 6")

### ---> check if predicted distances for multiple frames are calculated correctly

# for this test using a different rnnMode, doesn't matter for what is checked
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="conv", debug=True).cuda()

featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)


print("--------------------------------------------------")

print("---> 7")

### ---> check if predicted distances for multiple frames are calculated correctly
###      with a different length mapping from LSTM - not to [0,1], but to [0.25,0.75]

# for this test using a different rnnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4, mode="simple", 
                                    map01range=(0.25,0.75), debug=True).cuda()

# 2nd dim are predLengths - like with option to pass detached lengths to predictors
featc = torch.tensor([[[1,-0.6],[2,-0.6],[3,-0.6],[4,-0.6],[5,-0.6],[6,-0.6],[7,0.4]],
                        [[1,-0.6],[2,-0.6],[3,-0.4],[4,-0.2],[5,0.],[6,-0.6],[7,-0.6]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors
#candidates = torch.zeros_like(featc).view(1,featc.shape[0],1,featc.shape[1],featc.shape[2]).repeat(3,1,5,1,1).cuda()

print(featc.shape)  #, candidates.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-3,:], predLengths)  #candidates[:,:,:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 8")

### ---> check if predicted distances for multiple frames are calculated correctly
# unused mode I tried 

pred = TimeAlignedPredictionNetwork(2, 2, 3, 3, rnnMode='LSTM', dropout=False, sizeInputSeq=5, mode="predStartDep",
    debug=True).cuda()

featc = torch.tensor([[[1,-0.6,-0.6],[1,-0.6,-0.6],[1,-0.6,-0.6],[1,-0.6,-0.6],[1,-0.6,-0.6],[1,0.4,-0.6],[1,0.4,0.4]],
                        [[1,-0.6,-0.6],[1,-0.6,-0.6],[1,-0.6,-0.4],[1,-0.6,-0.2],[1,-0.6,0.],[1,-0.6,-0.6],[1,-0.6,-0.6]]]).cuda()
predLengths = featc[:,:,-2:]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths before mapping:", predLengths)  # this is BEFORE mapping!

pred(featc[:,:-2,:], predLengths)  # featc[:,:,-2:])  featc[:,:,-1])