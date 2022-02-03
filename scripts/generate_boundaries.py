import numpy as np
import os
import tqdm

def delta_BIC(feature_segment, i_min=0, lambda_=0):
    """
    Calculates the BIC for a given frame of audio.

    features_segment : CPC_feature of shape (frames, embedding_dim)
    i_min : i_min < i < n-i_min within the frame. n is segment_length
    lambda_ : hyperparameter lambda
    
    Returns delta BIC for i_min < i < n-i_min 
    """
    n, embedding_dim = feature_segment.shape
    scores=[]
    cov = np.linalg.norm(np.cov(feature_segment))
    for i in range(i_min, n-i_min):
        cov_x = np.linalg.norm(np.cov(feature_segment[:i]))
        cov_y = np.linalg.norm(np.cov(feature_segment[i:]))
        bic = (n/2)*np.log(cov) - (i/2)*np.log(cov_x) - ((n - i)/2)*np.log(cov_y)\
              - 0.5*lambda_*(embedding_dim + 0.5*embedding_dim*(embedding_dim + 1))*np.log(n)
        scores.append(bic)
    return scores

if __name__ == '__main__':
    pass

