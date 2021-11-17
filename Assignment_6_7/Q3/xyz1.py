import numpy as np

A = np.array([[0,1],
              [-4,-4]])

B = np.array([[0],[1]])
C = np.matmul(np.matmul(A,A),B)
print(np.linalg.matrix_rank(C))
print(C)