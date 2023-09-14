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
        # Matrix that checks if the set of images are linearly independant
        vectors = images.reshape(images.shape[0], -1)
        num_vectors = len(vectors)
        self.gram = np.zeros((num_vectors, num_vectors))
        for i, vector_i in enumerate(vectors):
            for j, vector_j in enumerate(vectors):
                self.gram[i, j] = np.dot(vector_i,  vector_j)

    def ModifiedGramSchmidt(self):
        images = self.loader.gray_flattened
        num_vectors = images.shape[0]
        num_dims = images.shape[1]

        # Calculate the norm of each row of the image matrix
        L = np.zeros(num_vectors)
        for i in range(num_vectors):
            L[i] = np.sqrt(np.dot(images[i], images[i]))

        # Normalise each row
        V = images.copy() / L
        B = V.copy()
        for j in range(0, num_vectors):
            B[j] = V[j] / np.sqrt(np.dot(V[j], V[j]))
            for k in range(j, num_vectors):
                V[k] = V[k] - np.dot(B[j], V[k])*B[j]

        return B
    
    def QRDecomposition(self, images):
        """
        Applies the Gram-Schmidt method to A
        and returns Q and R, so Q*R = A.
        """
        R = np.zeros((images.shape[1], images.shape[1]))
        Q = np.zeros(images.shape)
        for k in range(0, images.shape[1]):
            R[k, k] = np.sqrt(np.dot(images[:, k], images[:, k]))
            Q[:, k] = images[:, k]/R[k, k]
            for j in range(k+1, images.shape[1]):
                R[k, j] = np.dot(Q[:, k], images[:, j])
                images[:, j] = images[:, j] - R[k, j]*Q[:, k]
        return Q, R

    def HouseHolderQRDecomposition(self):
        images = self.loader.gray_flattened.T
        [m,n] = images.shape
        Q = np.eye(m)
        R = images.copy()

        for j in range(1, n):
            print(j)
            normx = np.norm(R[j:-1, j])
            s = -np.sign(R[j,j])
            u1 = R[j,j] - s * normx
            w = R[j:-1, j] / u1
            w[1] = 1
            tau = -s * u1 / normx

            R[j:-1, :] = R[j:-1, :] - (tau * w) @ (w.T * R[j:-1, :])
            Q[:, j:-1] = Q[:, j:-1] - (Q[:, j : -1] * w) @ (tau * w)
        
        return Q, R


    
    
    
    # def LinearDecompositionGhosts(self):
    #     '''
    #     Perform a linear recomposition of PCA components with the ghosts
    #     ''' 
    #     self.separator = SourceSeparator()
    #     self.separator.PerformTransform(self.loader.fftmag)

    #     results = np.zeros((len(self.constructor.ghosts), self.separator.PCA_images.shape[0] ))
    #     norms = np.zeros(len(self.constructor.ghosts))
    #     for i, Ghost in enumerate(self.constructor.ghosts):
    #         for j, image in enumerate(self.separator.PCA_images):
    #             if Ghost.fail == False:
    #                 results[i, j] = np.dot(image.reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(image.reshape(-1), image.reshape(-1))
    #                 norms[i] = np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))

    #     sums = np.sum(np.square(results), 1)
         
        
        
    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (results)
    #     ## save to csv file
    #     filepath = 'decomposition.csv'
    #     df.to_csv(filepath, index=False)
        
    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (sums)
    #     ## save to csv file
    #     filepath = 'decompositionsum.csv'
    #     df.to_csv(filepath, index=False)
        
    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (norms)
    #     ## save to csv file
    #     filepath = 'decompositionnorms.csv'
    #     df.to_csv(filepath, index=False)
        
    
    # def LinearDecompositionImages(self):
    #     '''
    #     Perform a linear recomposition of PCA components with the ghosts
    #     ''' 
    #     self.separator = SourceSeparator()
    #     self.separator.PerformTransform(self.loader.fftmag)

    #     results = np.zeros((self.separator.PCA_images.shape[0], len(self.constructor.ghosts)))
    #     norms = np.zeros(self.separator.PCA_images.shape[0])
        
    #     for i, image in enumerate(self.separator.PCA_images):
    #         for j, Ghost in enumerate(self.constructor.ghosts):
    #             if Ghost.fail == False:
    #                 results[i, j] = np.dot(image.reshape(-1), Ghost.ghost.reshape(-1)) / np.dot(Ghost.ghost.reshape(-1), Ghost.ghost.reshape(-1))
    #                 norms[i] = np.dot(image.reshape(-1), image.reshape(-1))

    #     sums = np.sum(np.square(results), 1)

    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (results)
    #     ## save to csv file
    #     filepath = 'decomposition.csv'
    #     df.to_csv(filepath, index=False)
        
    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (sums)
    #     ## save to csv file
    #     filepath = 'decompositionsum.csv'
    #     df.to_csv(filepath, index=False)
        
    #     ## convert your array into a dataframe
    #     df = pd.DataFrame (norms)
    #     ## save to csv file
    #     filepath = 'decompositionnorms.csv'
    #     df.to_csv(filepath, index=False)
         


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
    ghost_transform.InitaliseGhosts(3)
    print(ghost_transform.loader.gray_flattened.shape)
    print(ghost_transform.ModifiedGramSchmidt())
    #ghost_transform.LinearDecompositionGhosts()

    