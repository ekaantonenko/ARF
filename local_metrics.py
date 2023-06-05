import numpy as np

def accuracy_individual(A, B):
    numerator = A[(A==B)].size
    denominator = A.size
    return numerator / denominator
        
def accuracy_joint(A,B):
    numerator = np.all(A==B, axis=1).sum()
    denominator = A.shape[0]
    return numerator / denominator
