from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from DataLoader import DataLoader

class SourceSeparator:
    """
    Class with functions to separate sets of images into PCA and ICA components
    """
    def __init__(self):
        """
        Source Separator Constructor
        """
        self.transformer_ica= FastICA(random_state=42)
        self.transformer_pca = PCA(random_state=42)

    #   @fn PerformTransform (0)
    def PerformTransform(self, images, num_images):
        """
        Perform PCA and ICA on a set of images
        @param images np.array([num_images : height : width : channels])
        @param num_images (int) : The number of images from the dataset that you want to perform PCA and ICA on
        """
        image_height = images.shape[1]
        image_width = images.shape[2]
        image_size = image_width * image_height
        
        scaler = StandardScaler()
        data_unscaled = np.copy(images[0:num_images, :, :].reshape(num_images, -1))
        data_scaled = scaler.fit_transform(images[0:num_images, :, :].reshape(num_images, -1))

        self.PCA = self.transformer_pca.fit(data_unscaled)
        self.PCA_images = np.copy(self.PCA.components_.reshape((image_size, image_height, image_width)))

        self.PCA_scaled = self.transformer_pca.fit(data_scaled)
        self.PCA_scaled_images = np.copy(self.PCA_scaled.components_.reshape((image_size, image_height, image_width)))

        #self.transformer_ica = FastICA(random_state=42, whiten='unit-variance', max_iter= 1)
        #self.ICA = self.transformer_ica.fit(data_unscaled)
        #self.ICA_images = self.ICA.components_.reshape((image_size, image_height, image_width))
        
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
    
    loader = DataLoader()
    loader.LoadTarfile(tar, files)
    loader.TripleChannelUnflatten()
    loader.FourierTransformImages()

    separator = SourceSeparator()

    # print(separator.labels)

    # #ica.DisplayImages(ica.gray)
    # #ica.DisplayImages(ica.fftmag, True)
    separator.PerformTransform(loader.fftmag)
    #separator.DisplayImages(separator.ICA_images)
    separator.DisplayImages(separator.PCA_images)
    # #print(ica.gray.shape)
    
    # # ica.dataset = ica.dataset[0:20, :]
   
    # # print(ica.ICA)
    

    
