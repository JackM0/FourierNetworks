import numpy as np

from DataLoader import DataLoader
from GhostConstructor import GhostCreator
from SourceSeparator import SourceSeparator
from scipy import linalg
import matplotlib.pyplot as plt

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
    
    def InitaliseGhosts(self, element_size, max_occurances = 2):
        """
        Initalise a set of ghosts to be used as the basis functions for the images fourier space
        """
        self.constructor = GhostCreator(32)
        self.constructor.CreateGhostsFrom2PSE(element_size,  max_occurances)
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
        print(linalg.det(self.gram))
        
    def ModifiedGramSchmidt(self):
        images = self.constructor.ghost_images.reshape((-1, 32 * 32))
        num_vectors = images.shape[0]
        num_dims = images.shape[1]

        # Calculate the norm of each row of the image matrix
         # Normalise each row
        L = np.zeros(num_vectors)
        V = images.copy()
        for i in range(num_vectors):
            L[i] = np.sqrt(np.dot(images[i], images[i]))
            V[i] = V[i] / L[i]

       
        B = V.copy()
        for j in range(0, num_vectors):
            print(j)
            B[j] = V[j] / np.sqrt(np.dot(V[j], V[j]))
            for k in range(j, num_vectors):
                V[k] = V[k] - np.dot(B[j], V[k])*B[j]

        return B

    def HouseHolderQRDecomposition(self):
        images = self.constructor.ghost_images.reshape((-1, 32 * 32)).T
        
        Q, R, P = linalg.qr(images, pivoting=True)
                        
        self.DisplayQR(Q)
        
        self.basis = Q.T.reshape((-1, 32 * 32))
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


    def DisplayQR(self, Q):
        print(Q.shape)
        Q[np.abs(Q) < 10**(-14)] = 0
        fig=plt.figure(figsize=(32, 32))
        num_columns = Q.shape[1]
        for i in range(64):
            ax=fig.add_subplot(8,8, i+1)
            ax.imshow(Q[:, i].reshape((32, 32)), cmap=plt.cm.bone)
        plt.show()
    
    
    def LinearDecompositionGhosts(self):
        '''
        Perform a linear recomposition of PCA components with the ghosts
        ''' 
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag)

        results = np.zeros((self.basis.shape[0], self.separator.PCA_images.shape[0]))
        norms = np.zeros(self.basis.shape[0])
        for i, basis in enumerate(self.basis):
            for j, image in enumerate(self.separator.PCA_images):
                    results[i, j] = np.dot(image.reshape(-1), basis.reshape(-1)) / np.dot(basis.reshape(-1), basis.reshape(-1))
                    norms[i] = np.dot(basis.reshape(-1), basis.reshape(-1))
        print(results)
        sums = np.sum(np.square(results), 1)
        print(sums)
         


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
    Q, R = ghost_transform.HouseHolderQRDecomposition()
    ghost_transform.ConstructGramMatrix(Q.T.reshape((-1,32, 32)))
    ghost_transform.LinearDecompositionGhosts()
    

    