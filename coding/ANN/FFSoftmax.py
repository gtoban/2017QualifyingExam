import numpy as np

z1 = np.tanh(1+2)
z2 = np.tanh(1)
a1 = z2
a2 = z1+z2
Z = (np.exp(a1) + np.exp(a2))
y1 = np.exp(a1)/Z
y2 = np.exp(a2)/Z

print(y1,' ',y2)
print("\nMatrixForm\n")
X = np.array([[1,2]])
W1 = np.array([[1,1],[1,0]])
W2 = np.array([[0,1],[1,1]])
Z = np.tanh(X.dot(W1))
A = Z.dot(W2)
expA = np.exp(A)
Y = expA/expA.sum(axis=1,keepdims=True)
print( Y)

