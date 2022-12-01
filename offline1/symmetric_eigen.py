import numpy as np

# n: dimensions of matrix
n = int(input())

# A: n x n invertible symmetric matrix of random integers 
A = np.random.randint(50, size=(n, n))
A = A + A.T     # make A symmetric

A_inv = np.linalg.inv(A)

while not np.allclose(np.dot(A, A_inv), np.eye(n)):
    print('hello')
    A = np.random.randint(50, size=(n, n))
    A = A + A.T
    A_inv = np.linalg.inv(A)
    
print(A)
# print(np.round(A_inv, 2))
# print(np.round(np.dot(A, A_inv), 2))

# Eigen Decomposition using NumPy’s library function 
eigen_values, eigen_vectors = np.linalg.eig(A)

print()
print('eigen values:')
print(eigen_values)
print()
print('eigen vectors:')
print(eigen_vectors)

# Reconstruct A from eigenvalue and eigenvectors

# Theory---------------------------------------------------------#
#                                                                #
#   eigen_vectors matrix of symmetric A is orthogonality         #
#   i.e., eigen_vectors_inv = eigen_vectors.T                    #
#                                                                #
#-----------------------------------------------------------------

re_A = np.matmul(eigen_vectors, np.matmul(np.diag(eigen_values), eigen_vectors.T))

print()
print("reconstructed A:")
print(re_A)
# print(np.round(re_A))
# print(np.real(np.round(re_A)))

# Check if the reconstruction is perfect
print('Check if the reconstruction is perfect :', np.allclose(A, re_A))