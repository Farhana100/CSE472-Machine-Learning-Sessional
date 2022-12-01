import numpy as np

# n, m: dimensions of matrix
n, m = [int(x) for x in input().split()]

# A: n x m matrix of random integers 
A = np.random.randint(50, size=(n, m))
print(A)

# Singular Value Decomposition using NumPy’s library function

u, d, vh = np.linalg.svd(A, full_matrices=True)

# print(u.shape)
# print(d.shape)
# print(vh.shape)

print()
print('singular values: ')
print(d)

print()
print('left-singular vectors: ')
print(u)

print()
print('right-singular vectors: ')
print(vh.T)


# # Reconstruct A from eigenvalue and eigenvectors

# # diagonal D
# D = np.diag(np.pad(d, (0, np.abs(n-m)), 'constant', constant_values=(0, 0)))[:n, :m]

# re_A = np.matmul(u, np.matmul(D, vh))
# print(re_A)


# Calculating the Moore-Penrose Pseudoinverse using NumPy’s library function
A_pinv_np = np.linalg.pinv(A)

print()
print('Moore-Penrose Pseudoinverse of A by numpy:')
print(A_pinv_np)


# print()
# print('check')
# print(np.matmul(A, A_pinv))
# print()
# print(np.matmul(A_pinv, A))


# Calculating the Moore-Penrose Pseudoinverse again using Eq. 2.47 
D_pinv = np.diag(np.pad( [1/x if x != 0 else x for x in d]  , (0, np.abs(n-m)), 'constant', constant_values=(0, 0)))[:m, :n]

# print()
# print("check D_pinv:")
# print(D_pinv)
# print(D)


A_pinv = np.matmul(vh.T, np.matmul(D_pinv, u.T))

print()
print('Moore-Penrose Pseudoinverse of A by equation:')
print(A_pinv)

# Checking if these two inverses are equal
print()
print('Check if these two inverses are equal: ', np.allclose(A_pinv, A_pinv_np))











