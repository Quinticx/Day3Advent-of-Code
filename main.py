import numpy as np
import scipy.stats as sp

def split(word):
    return [char for char in word]

sweeps = np.loadtxt('input.txt', dtype=str)

splitted = np.zeros((1000, 12))

for i,diag in enumerate(sweeps):
   splitted[i] = split(sweeps[i])

most = sp.mode(splitted, axis=0) 

gamma = most.mode
gamma = gamma.astype("int")
print("Gamma: ",gamma)

gamma = gamma.astype("bool")

epsilon = np.invert(gamma, dtype=bool)
epsilon = epsilon.astype("int")
print("Epsilon: ", epsilon)

# Convert binary to int via 2 powered range array and do dot product
gamma = gamma.dot(2**np.arange(gamma.size)[::-1])
print(gamma)

epsilon = epsilon.dot(2**np.arange(epsilon.size)[::-1])
print(epsilon)

total = gamma*epsilon
print(total)


