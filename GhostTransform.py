import numpy as np

from DataLoader import DataLoader
from GhostConstructor import GhostCreator
from SourceSeparator import SourceSeparator

import pandas as pd

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
    
    def ConstructGramMatrix(self, images):
        vectors = images.reshape(images.shape[0], -1)
        num_vectors = len(vectors)
        self.gram = np.zeros((num_vectors, num_vectors))
        for i, vector_i in enumerate(vectors):
            for j, vector_j in enumerate(vectors):
                self.gram[i, j] = np.dot(vector_i,  vector_j)
        
        ## convert your array into a dataframe
        df = pd.DataFrame(self.gram)
        ## save to csv file
        filepath = 'grammatrix.csv'
        df.to_csv(filepath, index=False)
    
    def gramschmidt(A):
        """
        Applies the Gram-Schmidt method to A
        and returns Q and R, so Q*R = A.
        """
        R = np.zeros((A.shape[1], A.shape[1]))
        Q = np.zeros(A.shape)
        for k in range(0, A.shape[1]):
            R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
            Q[:, k] = A[:, k]/R[k, k]
            for j in range(k+1, A.shape[1]):
                R[k, j] = np.dot(Q[:, k], A[:, j])
                A[:, j] = A[:, j] - R[k, j]*Q[:, k]
        return Q, R
    
    
    
    def LinearDecompositionGhosts(self):
        '''
        Perform a linear recomposition of PCA components with the ghosts
        ''' 
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag)

        results = np.zeros((len(self.constructor.ghosts), self.separator.PCA_images.shape[0] ))
        norms = np.zeros(len(self.constructor.ghosts))
        for i, Ghost in enumerate(self.constructor.ghosts):
            for j, image in enumerate(self.separator.PCA_images):
                if Ghost.fail == False:
                    results[i, j] = np.dot(image.reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(image.reshape(-1), image.reshape(-1))
                    norms[i] = np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))

        sums = np.sum(np.square(results), 1)
         
        
        
        ## convert your array into a dataframe
        df = pd.DataFrame (results)
        ## save to csv file
        filepath = 'decomposition.csv'
        df.to_csv(filepath, index=False)
        
        ## convert your array into a dataframe
        df = pd.DataFrame (sums)
        ## save to csv file
        filepath = 'decompositionsum.csv'
        df.to_csv(filepath, index=False)
        
        ## convert your array into a dataframe
        df = pd.DataFrame (norms)
        ## save to csv file
        filepath = 'decompositionnorms.csv'
        df.to_csv(filepath, index=False)
        
    
    def LinearDecompositionImages(self):
        '''
        Perform a linear recomposition of PCA components with the ghosts
        ''' 
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag)

        results = np.zeros((self.separator.PCA_images.shape[0], len(self.constructor.ghosts)))
        norms = np.zeros(self.separator.PCA_images.shape[0])
        
        for i, image in enumerate(self.separator.PCA_images):
            for j, Ghost in enumerate(self.constructor.ghosts):
                if Ghost.fail == False:
                    results[i, j] = np.dot(image.reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))
                    norms[i] = np.dot(image.reshape(-1), image.reshape(-1))

        sums = np.sum(np.square(results), 1)

        ## convert your array into a dataframe
        df = pd.DataFrame (results)
        ## save to csv file
        filepath = 'decomposition.csv'
        df.to_csv(filepath, index=False)
        
        ## convert your array into a dataframe
        df = pd.DataFrame (sums)
        ## save to csv file
        filepath = 'decompositionsum.csv'
        df.to_csv(filepath, index=False)
        
        ## convert your array into a dataframe
        df = pd.DataFrame (norms)
        ## save to csv file
        filepath = 'decompositionnorms.csv'
        df.to_csv(filepath, index=False)
         


    


    


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
    ghost_transform.ConstructGramMatrix(ghost_transform.constructor.ghost_images)
    #ghost_transform.LinearDecompositionGhosts()

    