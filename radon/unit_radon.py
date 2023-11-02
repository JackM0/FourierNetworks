'''
Testss for discrete Radon transforms
'''
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# from transforms import Fourier, NormalizeStatistics
import radon

# Hyper-parameters
plot_size = 4
N = 28
Nmax = 37
# nmax = Nmax//2
# crop_size = 29
aperiodic_order = True

#network IO paths
path = 'F:/'

#--------------
#Data
print("> Setup dataset")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,)), transforms.Pad([0, 0, Nmax-N, Nmax-N], fill=0, padding_mode='constant')])
trainset = torchvision.datasets.MNIST(root=path+'data/mnist', train=True, download=True, transform=transform)

# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(N, padding=4, padding_mode='reflect'),
#     transforms.Pad([0, 0, Nmax-N, Nmax-N], fill=0, padding_mode='constant'),
#     NormalizeStatistics(unit_variance=False), #zero DC
#     Fourier(center=True, filter=filter),
#     transforms.CenterCrop(crop_size),
# ])
# trainset = torchvision.datasets.CIFAR10(root=path+'data/cifar10', train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True) #num_workers=6

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#Radon transform as a PyTorch network
frt = radon.fdrt(Nmax, mojette=aperiodic_order)
ifrt = radon.fidrt(Nmax, mojette=aperiodic_order)

print("Processing data ...")
start_time = time.time()
Rx = 0
X = 0
Ry = 0
for i, (x, y) in enumerate(train_loader):
    print(x.shape)
    print(x.dtype)
    Rx = frt(x)
    X = ifrt(Rx)
    Ry = y
    break
epoch_time = time.time() - start_time
print("Took " + str(epoch_time) + " secs or " + str(epoch_time/60) + " mins in total")

#projection power spects as images
plt.figure(figsize=(10, 10))

for i in range(plot_size**2):
    # define subplot
    plt.subplot(plot_size, plot_size, 1 + i)
    # turn off axis
    plt.axis('off')
    plt.tight_layout() #reduce white space
    # plot raw pixel data
    plt.imshow(torch.real(Rx[i,:,:]), cmap='gray')
    plt.title("Sinogram "+str(Ry[i]))
    
#projection phase as images
plt.figure(figsize=(10, 10))

for i in range(plot_size**2):
    # define subplot
    plt.subplot(plot_size, plot_size, 1 + i)
    # turn off axis
    plt.axis('off')
    plt.tight_layout() #reduce white space
    # plot raw pixel data
    plt.imshow(torch.real(X[i,:,:]), cmap='gray')
    plt.title("Reconstruct "+str(Ry[i]))

plt.show()
print("END")
