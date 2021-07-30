

from cpc.model import CPCModel
from copy import deepcopy
import torch

class EncFake:
    def __call__(self, x):
        return x.permute(0,2,1)  
        # ^ as Encoder gives weird-dim-order output, CPC model forward permutes it same way
    def getDimOutput(self):
        return 2


encFake = EncFake()  #lambda x: x.permute(0,2,1)
ARFake = lambda x: x

modSettingsBase = {
    "modDebug": True,
    "numProtos": 3, 
    "pushLossWeightEnc": 0.1,
    "pushLossWeightCtx": None,
    "VQpushEncCenterWeightOnTopConv": None,
    "VQpushEncCenterWeightOnlyAR": None,
    "VQpushEncCenterWeightOnlyCriterion": None,
    "VQgradualStart": None,
    "VQpushCtxCenterWeight": None,
    "pushLossLinear": True,  #
    "pushLossGradualStart": None,
    "pushLossProtosMult": None,
    "pushLossCenterNorm": False,  #
    "pushLossPointNorm": False,  #
    "pushLossNormReweight": None,  #
    "pushLossReweightPointsSeparately": False,  #
    "hierARshorten": None,
    "hierARgradualStart": None,
    "hierARmergePrior": None,
    "modelLengthInARsimple": False,
    "modelLengthInARconv": None,
    "modelLengthInARpredDep": None,
    "showLengthsInCtx": True,
    "shrinkEncodingsLengthDims": False
}

epochNrs = (2,50)

print("--------------------------------------------")
print("--> 1")

modSettings1 = deepcopy(modSettingsBase)

cpcModel1 = CPCModel(encFake, ARFake, 32, modSettings1)

givenCenters1 = torch.tensor([[1.5,1.5], [2.,2.], [5.,5.]], dtype=torch.float).cuda()
encoded_data1 = torch.tensor([[[1.,1.], [2.,2.]], [[3.,3.], [8.,9.]]], dtype=torch.float).cuda()
# closest centroids (Euclid) & lin distances: 0 (sqrt(2)/2), 1 (0.), 1 (sqrt(2)), 2 (5)
# sum dist: 1.5 sqrt(2) + 5 = 7.1213
# loss 1/40 of that (weight 0.1 and 4 points - it's avg): 0.1780
c_feature1 = 2. * encoded_data1
c_feature1_2 = c_feature1.clone()
encoded_data1_2 = encoded_data1.clone()
pushLoss1, closestCountsDataPar1, c_feature1_r, encoded_data1_r = \
    cpcModel1(c_feature1, encoded_data1, c_feature1_2, encoded_data1_2, givenCenters1, epochNrs, True, False)

print("--------------------------------------------")
print("--> 2")

modSettings2 = deepcopy(modSettingsBase)
modSettings2["pushLossCenterNorm"] = True
modSettings2["pushLossPointNorm"] = True
modSettings2["pushLossNormReweight"] = True

cpcModel2 = CPCModel(encFake, ARFake, 32, modSettings2)

givenCenters2 = torch.tensor([[1.,1.], [1.,-1.], [-2.,1.]], dtype=torch.float).cuda()
# after norm: [1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)], [2 / sqrt(5), 1 / sqrt(5)]
#             = [0.7071, 0.7071], [0.7071, -0.7071], [-0.8944, 0.4472]
encoded_data2 = torch.tensor([[[1.,1.5], [2.,-2.5]], [[-4.,2.], [4.,4.]]], dtype=torch.float).cuda()
# lengths (approx): 1.8028, 3.2016, 4.4721, 5.6569
# after norm (approx): [0.5547,0.8320], [0.6247,-0.7809], [-0.8944,0.4472], [0.7071,0.7071]
# closest after norm (diff; Euclid dist): 0 ([-0.1524,0.1249]; 0.1970), 1 ([-0.0824,-0.0738]; 0.1106), 2 ([0,0]; 0), 0 ([0,0]; 0)
# avg encoded_data length: 3.78335
# sum dist reweighted: (0.1970 + 0.1106) * 3.78335 = 1.1638, 
# loss 1/40 of that (weight 0.1 and 4 points - it's avg): 0.0291
c_feature2 = 2. * encoded_data2
c_feature2_2 = c_feature2.clone()
encoded_data2_2 = encoded_data2.clone()
pushLoss2, closestCountsDataPar2, c_feature2_r, encoded_data2_r = \
    cpcModel2(c_feature2, encoded_data2, c_feature2_2, encoded_data2_2, givenCenters2, epochNrs, True, False)

print("--------------------------------------------")
print("--> 3")

modSettings3 = deepcopy(modSettingsBase)
modSettings3["pushLossCenterNorm"] = True
modSettings3["pushLossPointNorm"] = True
modSettings3["pushLossNormReweight"] = True
modSettings3["pushLossReweightPointsSeparately"] = True

cpcModel3 = CPCModel(encFake, ARFake, 32, modSettings3)

givenCenters3 = torch.tensor([[1.,1.], [1.,-1.], [-2.,1.]], dtype=torch.float).cuda()
# after norm: [1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)], [2 / sqrt(5), 1 / sqrt(5)]
#             = [0.7071, 0.7071], [0.7071, -0.7071], [-0.8944, 0.4472]
encoded_data3 = torch.tensor([[[1.,1.5], [2.,-2.5]], [[-4.,2.], [4.,4.]]], dtype=torch.float).cuda()
# lengths (approx): 1.8028, 3.2016, 4.4721, 5.6569
# after norm (approx): [0.5547,0.8320], [0.6247,-0.7809], [-0.8944,0.4472], [0.7071,0.7071]
# closest after norm (diff; Euclid dist): 0 ([-0.1524,0.1249]; 0.1970), 1 ([-0.0824,-0.0738]; 0.1106), 2 ([0,0]; 0), 0 ([0,0]; 0)
# sum dist reweighted separately: 0.1970 * 1.8028 + 0.1106 * 3.2016 = 0.7092, 
# loss 1/40 of that (weight 0.1 and 4 points - it's avg): 0.0177
c_feature3 = 2. * encoded_data3
c_feature3_2 = c_feature3.clone()
encoded_data3_2 = encoded_data3.clone()
pushLoss3, closestCountsDataPar3, c_feature3_r, encoded_data3_r = \
    cpcModel3(c_feature2, encoded_data3, c_feature3_2, encoded_data3_2, givenCenters3, epochNrs, True, False)

print("--------------------------------------------")
print("--> 4")

modSettings4 = deepcopy(modSettingsBase)
modSettings4["pushLossCenterNorm"] = True
modSettings4["pushLossPointNorm"] = True
modSettings4["pushLossNormReweight"] = False

cpcModel4 = CPCModel(encFake, ARFake, 32, modSettings4)

givenCenters4 = torch.tensor([[1.,1.], [1.,-1.], [-2.,1.]], dtype=torch.float).cuda()
# after norm: [1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)], [2 / sqrt(5), 1 / sqrt(5)]
#             = [0.7071, 0.7071], [0.7071, -0.7071], [-0.8944, 0.4472]
encoded_data4 = torch.tensor([[[1.,1.5], [2.,-2.5]], [[-4.,2.], [4.,4.]]], dtype=torch.float).cuda()
# lengths (approx): 1.8028, 3.2016, 4.4721, 5.6569
# after norm (approx): [0.5547,0.8320], [0.6247,-0.7809], [-0.8944,0.4472], [0.7071,0.7071]
# closest after norm (diff; Euclid dist): 0 ([-0.1524,0.1249]; 0.1970), 1 ([-0.0824,-0.0738]; 0.1106), 2 ([0,0]; 0), 0 ([0,0]; 0)
# sum dist (approx): 0.3076, 
# loss 1/40 of that (weight 0.1 and 4 points - it's avg): 0.00769
c_feature4 = 2. * encoded_data4
c_feature4_2 = c_feature4.clone()
encoded_data4_2 = encoded_data4.clone()
pushLoss4, closestCountsDataPar4, c_feature4_r, encoded_data4_r = \
    cpcModel4(c_feature4, encoded_data4, c_feature4_2, encoded_data4_2, givenCenters4, epochNrs, True, False)

print("--------------------------------------------")
print("--> 5")

modSettings5 = deepcopy(modSettingsBase)
modSettings5["pushLossLinear"] = False

cpcModel5 = CPCModel(encFake, ARFake, 32, modSettings5)

givenCenters5 = torch.tensor([[1.5,1.5], [2.,2.], [5.,5.]], dtype=torch.float).cuda()
encoded_data5 = torch.tensor([[[1.,1.], [2.,2.]], [[3.,3.], [8.,9.]]], dtype=torch.float).cuda()
# closest centroids (Euclid) & sq distances: 0 (0.5), 1 (0.), 1 (2), 2 (25)
# sum dist: 27.5, 
# loss 1/40 of that (weight 0.1 and 4 points - it's avg): 0.6875
c_feature5 = 2. * encoded_data5
c_feature5_2 = c_feature5.clone()
encoded_data5_2 = encoded_data5.clone()
pushLoss5, closestCountsDataPar5, c_feature5_r, encoded_data5_r = \
    cpcModel5(c_feature5, encoded_data5, c_feature5_2, encoded_data5_2, givenCenters5, epochNrs, True, False)