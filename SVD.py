import numpy as np
matrix=np.array([[2,4],[3,8]])
U,S,VT=np.linalg.svd(matrix)

print("U matrix:")
print(U)
print("Singular values (as a diagonal matrix):")
print(np.diag(S))
print("VT matrix:")
print(VT)
print("Reconstruct Matrix ")
recon = np.dot(U, np.dot(np.diag(S), VT))
print(recon)
