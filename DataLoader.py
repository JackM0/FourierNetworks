import tarfile
import numpy as np
import os
import matplotlib.pyplot as plt

class DataLoader:

    def init(self):
        self.dataset = np.array([])

    def LoadTarfile(self, tar, files):
        with tarfile.open(tar) as tar_object:
            # Each file contains 10,000 color images and 10,000 labels
            fsize = 10000 * (32 * 32 * 3) + 10000

            # There are 6 files (5 train and 1 test)
            buffr = np.zeros(fsize * 6, dtype='uint8')

            # Get members of tar corresponding to data files
            # -- The tar contains README's and other extraneous stuff
            members = [file for file in tar_object if file.name in files]

            # Sort those members by name
            # -- Ensures we load train data in the proper order
            # -- Ensures that test data is the last file in the list
            members.sort(key=lambda member: member.name)

            # Extract data from members
            for i, member in enumerate(members):
                # Get member as a file object
                f = tar_object.extractfile(member)
                # Read bytes from that file object into buffr
                buffr[i * fsize:(i + 1) * fsize] = np.frombuffer(f.read(), 'B')

            # Parse data from buffer
            # -- Examples are in chunks of 3,073 bytes
            # -- First byte of each chunk is the label
            # -- Next 32 * 32 * 3 = 3,072 bytes are its corresponding image

            # Labels are the first byte of every chunk
            self.interger_labels = buffr[::3073]
            
            # Pixels are everything remaining after we delete the labels
            pixels = np.delete(buffr, np.arange(0, buffr.size, 3073))
            self.dataset = pixels.reshape(-1, 3072).astype('float32') / 255

    def TripleChannelUnflatten(self):
        self.channel_1 = self.dataset[:, 0 : 1024].reshape(60000, 32, 32)
        self.channel_2 = self.dataset[:, 1024 : 2 * 1024].reshape(60000, 32, 32)
        self.channel_3 = self.dataset[:, 2 * 1024 : 3 * 1024].reshape(60000, 32, 32)
        self.images = np.zeros((60000, 32, 32, 3))
        self.images[:, :, :, 0] = self.channel_1
        self.images[:, :, :, 1] = self.channel_2
        self.images[:, :, :, 2] = self.channel_3
        self.rgb2gray()
        
        self.gray_flattened = self.gray.reshape((60000, 32 * 32))
        
    def rgb2gray(self):
        r, g, b = self.channel_1, self.channel_2, self.channel_3
        self.gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    def FourierTransformImages(self):
        fft = np.fft.fftshift(np.fft.fft2(self.gray))
        self.fftmag = np.abs(fft)
        self.fftphase = np.angle(fft)