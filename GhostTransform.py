import numpy as np

from DataLoader import DataLoader
from GhostConstructor import GhostCreator
from SourceSeparator import SourceSeparator

class GhostTransform:
    """
    Class that will Create a Set of Ghosts using possible combinations of 2PSE
    """
    def __init__(self):
        self.data_images = np.array([])
        self.loader = DataLoader()
        return
    
    def LoadRGBImagesFromTarfile(self, tar, files):
        self.loader.LoadTarfile(tar, files)
        self.loader.TripleChannelUnflatten()
        self.loader.FourierTransformImages()
    
    def InitaliseGhosts(self, element_size):
        """
        Initalise a set of ghosts to be used as the basis functions for the images fourier space
        """
        self.constructor = GhostCreator(32)
        self.constructor.PossibleConvolutionGhosts(element_size)
        print("ghosts made")
        return
    
    def LinearDecomposition(self):
        '''
        Perform a linear recomposition of PCA components with the ghosts
        '''
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag)

        results = np.zeros(len(self.constructor.ghosts))

        for i, Ghost in enumerate(self.constructor.ghosts):
            if Ghost.fail == False:
                results[i] = np.dot(self.separator.PCA_images[1].reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))

        print(results)
        print(np.sum(np.square(results[1:])), np.sum(results[1:]), np.sum(np.square(self.separator.PCA_images[1].reshape(-1))))

        for i, Ghost in enumerate(self.constructor.ghosts):
            if Ghost.fail == False:
                results[i] = np.dot(self.separator.PCA_images[2].reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))

        print(results)
        print(np.sum(np.square(results[1:])), np.sum(results[1:]), np.sum(np.square(self.separator.PCA_images[2].reshape(-1))))

        for i, Ghost in enumerate(self.constructor.ghosts):
            if Ghost.fail == False:
                results[i] = np.dot(self.separator.PCA_images[3].reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))

        print(results)
        print(np.sum(np.square(results[1:])), np.sum(results[1:]), np.sum(np.square(self.separator.PCA_images[3].reshape(-1))))
            


    


    


if __name__ == '__main__':
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    
    ghost_transform = GhostTransform()
    ghost_transform.LoadRGBImagesFromTarfile(tar, files)
    ghost_transform.InitaliseGhosts(5)
    ghost_transform.LinearDecomposition()

    