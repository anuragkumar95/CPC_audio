


from cpc.model import seDistancesToCentroids
import torch 


batch = torch.tensor([[[1,2], [4,3], [5,6]], [[1,1], [3,3], [8,8]]], dtype=float)
centroids = torch.tensor([[0,0], [2,2], [2.5,3]], dtype=float)

# correctness test, on CPU

dists1 = seDistancesToCentroids(batch, centroids, debug=True)
print("SQ DISTS 1:", dists1)

# for this one, distances to centroid 0 will collapse as it can't be normalized
# so distances to this one will behave weird; should never occur in real training
dists2 = seDistancesToCentroids(batch, centroids, doNorm=True, debug=True)
print("SQ DISTS 2:", dists2)
