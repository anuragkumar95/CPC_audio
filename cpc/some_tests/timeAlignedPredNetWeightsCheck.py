

import torch
from cpc.criterion.soft_align import TimeAlignedPredictionNetwork


### lengths are 0.2 which after remapping is 0.6: to subsequent frames, 0.6, 1.2 1.8

print("---> 1")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 1.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 2")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 2.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 3")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("doubleExp", 1.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 4")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("bilin", None), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 5")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("trilin", None), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 6")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("normals", None), modelNormalsSettings=(0.2, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 7")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 1.), modelNormalsSettings=(0.2, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 8")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 3, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("trilin", None), modelNormalsSettings=(0.6, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")