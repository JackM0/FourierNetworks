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
    
    def gram_schmidt(self):
        vectors = self.constructor.ghost_images.reshape((-1, 32 * 32))
        num_vectors, vector_dim = vectors.shape
        ortho_basis = np.zeros((num_vectors, vector_dim))
        
        for i in range(num_vectors):
            print(i)
            # Start with the original vector
            ortho_basis[i] = vectors[i]
            
            # Subtract projections onto previously orthogonalized vectors
            for j in range(i):
                projection = np.dot(vectors[i], ortho_basis[j]) / np.dot(ortho_basis[j], ortho_basis[j])
                ortho_basis[i] -= projection * ortho_basis[j]
            
            # Normalize the orthogonalized vector
            norm = np.linalg.norm(ortho_basis[i])
            if norm != 0:
                ortho_basis[i] /= norm
        
        return ortho_basis
    
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
        images = self.constructor.ghost_images.reshape((-1, 32 * 32)).T
        print(images.shape)
        [m,n] = images.shape
        Q = np.eye(m)
        R = images.copy()

        for j in range(n):
            print(j)
            max_index = np.argmax(np.abs(R[j:, j])) + j
            print(max_index)
            if max_index != j:
                # Swap columns in both R and Q to make R[j,j] the largest element
                R[[j, max_index], :] = R[[max_index, j], :]
                Q[[j, max_index], :] = Q[[max_index, j], :]
            normx = np.sqrt(np.dot(R[j:, j], R[j:, j]))
            print(normx)
            s = -np.sign(R[j,j])
            print(s, R[j,j])
            u1 = R[j,j] - s * normx
            print(u1)
            w = (R[j:, j] / u1).reshape((-1, 1))
            #print(w)
            #print(w)
            w[0] = 1
            tau = -s * u1 / normx
            #print(tau.shape)
            #print('testmg')
            step1 = w.T @ R[j:, :]
            step2 = (tau * w)
            #print(step1.shape)
            #print(step2.shape)

            R[j:, :] = R[j:, :] - (tau * w) @ (w.T @ R[j:, :])
            Q[:, j:] = Q[:, j:] - (Q[:, j :] @ w) @ (tau * w).T
        
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
    # Q, R = ghost_transform.HouseHolderQRDecomposition()
    # print(Q)

    # print(R)
    print(ghost_transform.gram_schmidt())
    #ghost_transform.LinearDecompositionGhosts()

    