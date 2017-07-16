import numpy as np 

dataset = [[1,20,1], [2,21,0], [3,22,1]]
# SVD
P,D,Q = np.linalg.svd(dataset, full_matrices=0)

#PCA reduce to k dim and let initial n dim feature
# so here dim of dataset = 3x3, where 3 columns are features let us reduce to 3x2

M = np.dot(P[:,:2], np.dot(np.diag(D[:2]), Q[:2,:]))
print M