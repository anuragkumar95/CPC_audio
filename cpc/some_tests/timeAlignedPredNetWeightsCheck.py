

import torch
from cpc.criterion.soft_align import TimeAlignedPredictionNetwork
from torch.distributions import Normal
from math import sqrt

### lengths are 0.2 which after remapping is 0.6: to subsequent frames, 0.6, 1.2 1.8

## all cases predict 3 frames and use 4 predictors (0,1,2,3)

### ---> check if weights are calculated correctly and predictions weighted correctly
### correct unnormalized weights are listed below in each case,
### and correct prediction weighting can be checked based on following debug: 
### predictsPerPredictor, weights, predsWeighted appearing near each case's end

print("---> 1")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 1.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# 1st prediction: 0.5488, 0.6703, 0.2466, 0.0907
# 2nd prediction: 0.3012, 0.8187, 0.4493, 0.1653
# 3rd prediction: 0.1653, 0.4493, 0.8187, 0.3012

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 2")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 2.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# 1st prediction: 0.3012, 0.4493, 0.0608, 0.0082
# 2nd prediction: 0.0907, 0.6703, 0.2019, 0.0273
# 3rd prediction: 0.0273, 0.2019, 0.6703, 0.0907

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 3")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("doubleExp", 1.), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# 1st prediction: 0.4395, 0.6115, 0.0471, 0.00004
# 2nd prediction: 0.0983, 0.8014, 0.2936, 0.0064
# 3rd prediction: 0.0064, 0.2936, 0.8014, 0.0983

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 4")

### ---> check if weights are calculated correctly and predictions weighted correctly

# here just look at weights debug, there is no separate unnormalized debug as normalization is not necessary in this case

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("bilin", None), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# 1st prediction: 0.4, 0.6, 0, 0
# 2nd prediction: 0, 0.8, 0.2, 0
# 3rd prediction: 0, 0.2, 0.8, 0

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 5")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("trilin", None), debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# 1st prediction: 0.9, 1.1, 0.1, 0
# 2nd prediction: 0.3, 1.3, 0.7, 0
# 3rd prediction: 0, 0.7, 1.3, 0.3

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")


print("---> 6")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("normals", None), modelNormalsSettings=(0.2, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

dist1 = Normal(0.6, 0.2)
dist2 = Normal(1.2, sqrt(0.2**2 * 2))
dist3 = Normal(1.8, sqrt(0.2**2 * 3))
def stripeMass(distribution, l, r):
    return (distribution.cdf(torch.tensor(r, dtype=float)) - distribution.cdf(torch.tensor(l, dtype=float))).item()
print("correct weights (unnormalized)")
# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# standard deviations for predictions 1,2,3: 0.2, 0.4, 0.6
# 1st prediction: 
print(stripeMass(dist1,-0.5,0.5),stripeMass(dist1,0.5,1.5),stripeMass(dist1,1.5,2.5),stripeMass(dist1,2.5,3.5))
# 2nd prediction: 
print(stripeMass(dist2,-0.5,0.5),stripeMass(dist2,0.5,1.5),stripeMass(dist2,1.5,2.5),stripeMass(dist2,2.5,3.5))
# 3rd prediction: 
print(stripeMass(dist3,-0.5,0.5),stripeMass(dist3,0.5,1.5),stripeMass(dist3,1.5,2.5),stripeMass(dist3,2.5,3.5))
print("end correct weights")

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 7")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("exp", 1.), modelNormalsSettings=(0.5, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# standard deviations for predictions 1,2,3: 0.5, sqrt(0.5^2 * 2)=0.7071, sqrt(0.5^2 * 3)=0.866, 
#                                            which serve as exponent denominator multipliers
# 1st prediction (exp 0.5): 0.3012, 0.4493, 0.0608, 0.0082
# 2nd prediction (exp sqrt(0.5^2 * 2)=0.7071): 0.1832, 0.7536, 0.3226, 0.0784
# 3rd prediction (exp sqrt(0.5^2 * 3)=0.866): 0.1251, 0.397, 0.7938, 0.2502

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")

print("---> 8")

### ---> check if weights are calculated correctly and predictions weighted correctly

# for this test using a different rnMode, doesn't matter for what is checked
# predicting 2-dim encodings from 1-dim context representations and detached length predictions (2 dim after all)
pred = TimeAlignedPredictionNetwork(3, 4, 2, 2, rnnMode='LSTM', dropout=False, sizeInputSeq=4,
                                    mode="simple", weightMode=("trilin", None), modelNormalsSettings=(0.6, 1.),
                                    debug=True).cuda()

featc = torch.tensor([[[1,0.2],[2,0.2],[3,0.2],[4,0.2]]]).cuda()
predLengths = featc[:,:,-1]  # like with option to pass detached lengths to predictors

# correct weights (unnormalized):
# predictors #0, #1, #2, #3
# standard deviations for predictions 1,2,3: 0.6, sqrt(0.6^2 * 2)=0.8485, sqrt(0.6^2 * 3)=1.0392 
#                                            (equal to linear-weight distance the predictor is seen in)
# 1st prediction (weight radius 0.6): 0, 0.2, 0, 0
# 2nd prediction (weight radius sqrt(0.6^2 * 2)=0.8485): 0, 0.6845, 0.0485, 0
# 3rd prediction (weight radius sqrt(0.6^2 * 3)=1.0392): 0, 0.2392, 0.8392, 0

print(featc.shape)

print("predLengths:", predLengths)

pred(featc[:,:-3,:], predLengths)

print("--------------------------------------------------")