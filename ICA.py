from sklearn.decomposition import FastICA, PCA
import tarfile
import numpy as np
import os
import matplotlib.pyplot as plt

class SourceSeparator:

    def __init__(self):
        self.transformer_ica= FastICA(max_iter=300, random_state=42)
        self.transformer_pca = PCA()
        
    def LoadDataset(self, dataset):
        self.dataset = dataset

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

            def _onehot(integer_labels):
                """Return matrix whose rows are onehot encodings of integers."""
                n_rows = len(integer_labels)
                n_cols = integer_labels.max() + 1
                onehot = np.zeros((n_rows, n_cols), dtype='uint8')
                onehot[np.arange(n_rows), integer_labels] = 1
                return onehot

            self.labels = _onehot(self.interger_labels)


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

        return

    def FourierTransformImages(self):
        fft = np.fft.fftshift(np.fft.fft2(self.gray))
        self.fftmag = np.abs(fft)
        self.fftphase = np.angle(fft)
    
    def rgb2gray(self):
        r, g, b = self.channel_1, self.channel_2, self.channel_3
        self.gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    def PerformTransform(self):
        
        self.ICA = self.transformer_ica.fit(self.fftmag[0:2000, :, :].reshape(2000, 32 * 32))
        self.PCA = self.transformer_pca.fit(self.fftmag[0:2000, :, :].reshape(2000, 32 * 32))
        
        self.ICA_images = self.ICA.components_.reshape((1024, 32, 32))
        self.PCA_images = self.PCA.components_.reshape((1024, 32, 32))
        print("done")

    def DisplayImages(self, images, log = False):
        fig=plt.figure(figsize=(16, 16))
        for i in range(64):
            ax=fig.add_subplot(8, 8, i+1)
            if log:
                ax.imshow(np.log(images[i, :, :]), cmap=plt.cm.bone)
            else:
                ax.imshow(images[i, :, :], cmap=plt.cm.bone)
        plt.show()

if __name__ == '__main__':
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    
    separator = SourceSeparator()
    separator.LoadTarfile(tar, files)
    print(separator.labels)
    separator.TripleChannelUnflatten()
    separator.FourierTransformImages()
    #ica.DisplayImages(ica.gray)
    #ica.DisplayImages(ica.fftmag, True)
    separator.PerformTransform()
    separator.DisplayImages(separator.ICA_images)
    separator.DisplayImages(separator.PCA_images)
    #print(ica.gray.shape)
    
    # ica.dataset = ica.dataset[0:20, :]
   
    # print(ica.ICA)
    

    
