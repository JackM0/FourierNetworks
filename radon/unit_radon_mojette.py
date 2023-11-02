'''
Test the Mojette ordering of DRT projections
'''
import torch
import matplotlib.pyplot as plt

import radon #local module
import farey #local module

N = 101
aperiodic_order = True

#create angle set with Farey vectors
fareyVectors = farey.Farey()        
fareyVectors.compactOn()
fareyVectors.generateFiniteWithCoverage(N, L1Norm=False)
angles = fareyVectors.vectors
mValues = fareyVectors.finiteAngles

lines = radon.dpradon.slices.getFareySlicesCoordinates(N, angles=angles, center=True)

slices = torch.arange(1,N+1, dtype=torch.float32)

dftSpaceOrder = torch.zeros((N,N), dtype=torch.float32)
# dftSpaceOrder[lines[0, 0, :], lines[0, 1, :]] = slices
dftSpaceOrder[lines[:, 0, :], lines[:, 1, :]] = slices

slices = torch.arange(1,N+2, dtype=torch.float32)
slices = slices.repeat(N, 1)
slices = torch.transpose(slices, 0, 1)
print(slices)
dftSpaceLines = torch.zeros((N,N), dtype=torch.float32)
# dftSpaceLines[lines[0, 0, :], lines[0, 1, :]] = slices[0]
dftSpaceLines[lines[:, 0, :], lines[:, 1, :]] = slices[:]

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#projection power spects as images
plt.figure(figsize=(4, 4))

plt.axis('off')
plt.tight_layout() #reduce white space
plt.imshow(dftSpaceOrder)

#projection power spects as images
plt.figure(figsize=(4, 4))

plt.axis('off')
plt.tight_layout() #reduce white space
plt.imshow(dftSpaceLines)

plt.show()
print("END")