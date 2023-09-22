import numpy as np

from DataLoader import DataLoader
from GhostConstructor import GhostCreator
from SourceSeparator import SourceSeparator
from scipy import linalg
import matplotlib.pyplot as plt

class GhostTransform:
    """
    Class with functions that are used to try to find a set of basis images using Ghosts as a starting point and decomposs images using these basis images
    """
    def __init__(self, N):
        self.data_images = np.array([])
        self.N = N
        self.loader = DataLoader()
        self.constructor = GhostCreator(N)
    
    def LoadRGBImagesFromTarfile(self, tar, files, im_shape, images_per_file):
        self.loader.LoadTarfile(tar, files, im_shape, images_per_file)
        self.loader.TripleChannelUnflatten()
        self.loader.FourierTransformImages()
    
    def InitaliseGhosts(self, size_grid, num_octants, max_occurances):
        """
        Initalise a set of ghosts to be used as the basis functions for the images fourier space
        """
        self.constructor.CreateGhostsFrom2PSE(size_grid, num_octants, max_occurances)
        print("Created Ghosts")
        return
        
    def HouseHolderQRDecomposition(self):
        # Convert an Array of N * N images into an array where each column is a vector of length N * N
        image_matrix = self.constructor.ghost_images.reshape((-1, self.N * self.N)).T
        Q, R, P = linalg.qr(image_matrix, pivoting=True)
        
        self.basis_images = Q.T.reshape((-1, self.N, self.N))
        self.basis_images[np.abs(self.basis_images) < 10**(-14)]    
        
        # print(images.shape)
        # [m,n] = images.shape
        # Q = np.eye(m)
        # R = images.copy()

        # for j in range(n):
        #     print(j)
        #     max_index = np.argmax(np.abs(R[j:, j])) + j
        #     print(max_index)
        #     if max_index != j:
        #         # Swap columns in both R and Q to make R[j,j] the largest element
        #         R[[j, max_index], :] = R[[max_index, j], :]
        #         Q[[j, max_index], :] = Q[[max_index, j], :]
                
                
        #     normx = np.sqrt(np.dot(R[j:, j], R[j:, j]))
        #     print(normx)
        #     s = -np.sign(R[j,j])
        #     print(s, R[j,j])
        #     u1 = R[j,j] - s * normx
        #     print(u1)
        #     w = (R[j:, j] / u1).reshape((-1, 1))
        #     #print(w)
        #     #print(w)
        #     w[0] = 1
        #     tau = -s * u1 / normx
        #     #print(tau.shape)
        #     #print('testmg')
        #     step1 = w.T @ R[j:, :]
        #     step2 = (tau * w)
        #     #print(step1.shape)
        #     #print(step2.shape)

        #     R[j:, :] = R[j:, :] - (tau * w) @ (w.T @ R[j:, :])
        #     Q[:, j:] = Q[:, j:] - (Q[:, j :] @ w) @ (tau * w).T
        
        return Q, R


    def DisplayImages(self, images, images_to_display):
        num_images = len(images_to_display)
        display_rows_cols = int(num_images**0.5)

        fig=plt.figure(figsize=(self.N, self.N))        
        for i, image_index in enumerate(images_to_display):
            ax=fig.add_subplot(display_rows_cols, display_rows_cols, i+1)
            ax.imshow(images[image_index, :, :], cmap=plt.cm.bone)
            
        plt.show()
    
    
    def LinearDecompositionGhosts(self):
        '''
        Perform a linear recomposition of PCA components with the ghosts
        ''' 
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag, len(self.loader.fftmag))

        results = np.zeros((self.basis_images.shape[0], self.separator.PCA_images.shape[0]))
        norms = np.zeros(self.basis_images.shape[0])
        for i, basis in enumerate(self.basis_images):
            for j, image in enumerate(self.separator.PCA_images):
                    results[i, j] = np.dot(image.reshape(-1), basis.reshape(-1)) / np.dot(basis.reshape(-1), basis.reshape(-1))
                    norms[i] = np.dot(basis.reshape(-1), basis.reshape(-1))
        
        print(np.square(results))
        print(np.argmax(np.square(results), 0))
        print(np.argmax(np.square(results), 1))
        sums = np.sum(np.square(results), 1)
        print(sums)
      
    def ConstructGramMatrix(self, images):
        # Matrix that checks if the set of images are linearly independant
        vectors = images.reshape(images.shape[0], -1)
        num_vectors = len(vectors)
        self.gram = np.zeros((num_vectors, num_vectors))
        for i, vector_i in enumerate(vectors):
            for j, vector_j in enumerate(vectors):
                self.gram[i, j] = np.dot(vector_i,  vector_j)
                
        print(f"The Gram Matrix has a determinate of {linalg.det(self.gram)}")   

if __name__ == '__main__':
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    
    
    ghost_transform = GhostTransform(32)
    ghost_transform.LoadRGBImagesFromTarfile(tar, files, (32, 32, 3), 10000)
    ghost_transform.InitaliseGhosts(size_grid = 3, num_octants = 4, max_occurances = 2)
    print(ghost_transform.loader.gray_flattened.shape)
    
    Q, R = ghost_transform.HouseHolderQRDecomposition()
    images_to_display = np.arange(0, 64, 1, dtype=int)
    # images_to_display = np.arange(0, 1024, 16, dtype=int)
    ghost_transform.DisplayImages(ghost_transform.basis_images, images_to_display)
    
    ghost_transform.ConstructGramMatrix(ghost_transform.basis_images)
    ghost_transform.LinearDecompositionGhosts()
    

    