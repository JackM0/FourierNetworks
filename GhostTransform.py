import numpy as np

from DataLoader import DataLoader
from GhostConstructor import GhostCreator
from SourceSeparator import SourceSeparator
from HermiteConstructor import HermiteConstructor
from scipy import linalg
import matplotlib.pyplot as plt
import sys
import os 
from scipy.integrate import simps
from sklearn.decomposition import PCA
from skimage.transform import radon, rescale, iradon
from sklearn.preprocessing import StandardScaler
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
        ghost_path = './ghost_image_arrays/'
        ghost_file_name = 'ghosts_grid_' + str(size_grid) + 'octants_' + str(num_octants) + 'elementuses_' + str(max_occurances) + '.npy'
        rf_file_name = 'rf_grid_' + str(size_grid) + 'octants_' + str(num_octants) + 'elementuses_' + str(max_occurances) + '.npy'

        if not os.path.isfile(ghost_path + ghost_file_name):
            self.constructor.CreateGhostsFrom2PSE(size_grid, num_octants, max_occurances)
            np.save(ghost_path + ghost_file_name, self.constructor.ghost_images)
            np.save(ghost_path + rf_file_name, self.constructor.receptive_field_images)
            print("Created Ghosts")
        else:
            print("Loading Ghosts")
            self.constructor.ghost_images = np.load(ghost_path + ghost_file_name)
            self.constructor.receptive_field_images = np.load(ghost_path + rf_file_name)

        return
    
    def InitialiseRandomBigGhosts(self, size_grid, num_octants, num_ghosts, num_elements):
        self.constructor.CreateBigGhosts(size_grid, num_octants, num_ghosts, num_elements)
        
    def HouseHolderQRDecomposition(self):
        # Convert an Array of N * N images into an array where each column is a vector of length N * N
        receptive_field_matrix = self.constructor.receptive_field_images.reshape((-1, self.N * self.N)).T
        Q, R, P = linalg.qr(receptive_field_matrix, pivoting=True)
        self.fourierghost_basis_images = Q.T.reshape((-1, self.N, self.N))

        # Convert an Array of N * N images into an array where each column is a vector of length N * N
        ghost_matrix = self.constructor.ghost_images.reshape((-1, self.N * self.N)).T
        Q, R, P = linalg.qr(ghost_matrix, pivoting=True)
        self.ghost_basis_images = Q.T.reshape((-1, self.N, self.N))
        

        self.ghost_basis_images_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_fft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_noshift_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_1dshift_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)

        for i, image in enumerate(self.ghost_basis_images):
            self.ghost_basis_images_fft[i, :, :] = np.fft.fftshift(np.fft.fft2(image))
            self.ghost_basis_images_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(np.unwrap(image)))
            self.ghost_basis_images_noshift_ifft[i, :, :] = np.fft.ifft2(np.unwrap(image))
            #self.ghost_basis_images_1dshift_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(image, 0))
            


        self.fourierghost_basis_images_fft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_ifft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_noshift_fft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_shiftedrow = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_shiftedcol = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        for i, image in enumerate(self.fourierghost_basis_images):
            self.fourierghost_basis_images_fft[i, :, :] = np.fft.fftshift(np.fft.fft2(image))
            self.fourierghost_basis_images_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(image))
            self.fourierghost_basis_images_noshift_fft[i, :, :] = np.fft.fft2(image)
            self.fourierghost_basis_images_shiftedrow[i, :, :] = np.fft.fftshift(image, 0)
            self.fourierghost_basis_images_shiftedcol[i, :, :] = np.fft.fftshift(image, 1)

    def  QRWithRadonFlattening(self):
        
        # Take the radon transform of each of the ghosts
        theta = np.linspace(0., 180., max(self.constructor.ghost_images.shape[1:]), endpoint=False)
        self.ghost_sinograms = np.zeros((self.constructor.ghost_images.shape))
        print(self.constructor.ghost_images.shape)
        for i, ghost in enumerate(self.constructor.ghost_images):
            print(i)
            self.ghost_sinograms[i] = radon(ghost, theta=theta)
            
        # Now fourier transform each row of the sinogram, so that each row now is a radial slice of the ghosts fourier transform
        self.fourier_slices = np.zeros((self.ghost_sinograms.shape), dtype=np.complex_)
        for i, sinogram in enumerate(self.ghost_sinograms):
            self.fourier_slices[i] = np.fft.fftshift(np.fft.fft(sinogram, axis = 0), axes = 0)
        
        # Perform the QR on these sets of radial slices
        slice_matrix = self.fourier_slices.reshape((-1, self.N * self.N)).T
        Q, R, P = linalg.qr(slice_matrix, pivoting=True)
        self.fourier_slice_basis_images = Q.T.reshape((-1, self.N, self.N))

        self.ghost_basis_images = np.zeros((self.fourier_slice_basis_images.shape))
        for j, basis_image in enumerate(self.fourier_slice_basis_images):
            print(j)
            sinogram = np.fft.ifft(np.fft.ifftshift(basis_image, axes = 0), axis = 0)
            self.ghost_basis_images[j] = iradon(sinogram, theta=theta, filter_name='ramp')

        return
    
    def PCAOfGhosts(self):
        ghost_vectors = self.constructor.ghost_images.reshape((-1, self.N * self.N))

        # # Standardize the data (mean=0 and variance=1)
        # scaler = StandardScaler()
        # ghost_vectors_standardised = scaler.fit_transform(ghost_vectors)
        
        n_components = 64  # Set the number of components to keep
        pca = PCA(n_components=n_components)
        pca.fit(ghost_vectors)
        self.ghost_basis_images = np.copy(pca.components_)
        self.ghost_basis_images =self.ghost_basis_images.reshape((-1, self.N, self.N))

        
        fourierghost_vectors = self.constructor.receptive_field_images.reshape((-1, self.N * self.N))

        # # Standardize the data (mean=0 and variance=1)
        # scaler = StandardScaler()
        # fourierghost_vectors_standardised = scaler.fit_transform(fourierghost_vectors)
        
        n_components = 64  # Set the number of components to keep
        pca = PCA(n_components=n_components)
        pca.fit(fourierghost_vectors.real)
        self.fourierghost_basis_images = np.copy(pca.components_)
        self.fourierghost_basis_images =self.fourierghost_basis_images.reshape((-1, self.N, self.N))
        
        
        self.ghost_basis_images_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_fft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_noshift_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)
        self.ghost_basis_images_1dshift_ifft = np.zeros(self.ghost_basis_images.shape, dtype=np.complex_)

        for i, image in enumerate(self.ghost_basis_images):
            self.ghost_basis_images_fft[i, :, :] = np.fft.fftshift(np.fft.fft2(image))
            self.ghost_basis_images_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(np.unwrap(image)))
            self.ghost_basis_images_noshift_ifft[i, :, :] = np.fft.ifft2(np.unwrap(image))
            #self.ghost_basis_images_1dshift_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(image, 0))

        self.fourierghost_basis_images_fft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_ifft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_noshift_fft = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_shiftedrow = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        self.fourierghost_basis_images_shiftedcol = np.zeros(self.fourierghost_basis_images.shape, dtype=np.complex_)
        for i, image in enumerate(self.fourierghost_basis_images):
            self.fourierghost_basis_images_fft[i, :, :] = np.fft.fftshift(np.fft.fft2(image))
            self.fourierghost_basis_images_ifft[i, :, :] = np.fft.ifft2(np.fft.ifftshift(image))
            self.fourierghost_basis_images_noshift_fft[i, :, :] = np.fft.fft2(image)
            self.fourierghost_basis_images_shiftedrow[i, :, :] = np.fft.fftshift(image, 0)
            self.fourierghost_basis_images_shiftedcol[i, :, :] = np.fft.fftshift(image, 1)
        
        return

    def DisplayImages(self, images, images_to_display, display):
        num_images = len(images_to_display)
        display_rows_cols = int(num_images**0.5)

        fig=plt.figure(figsize=(self.N, self.N))        
        for i, image_index in enumerate(images_to_display):
            ax=fig.add_subplot(display_rows_cols, display_rows_cols, i+1)
            ax.imshow(images[image_index, :, :], cmap=plt.cm.bone)

        if display:  
            plt.show()

        return fig

    def SaveAllImages(self, images, name_prefix, location):
        if(not os.path.exists(location)):
            os.makedirs(location)

        num_images = images.shape[0]
        for i in range(1): #range(2)
            images_to_display = np.arange(0 + i * 64, (i + 1) * 64, 1, dtype=int)
            self.DisplayImages(images, images_to_display, display = False).savefig(location + '/' + name_prefix + str(i) + '.png')
            plt.close()
            if (num_images < 64 * 2):
                break
    
    def LinearDecompositionGhosts(self):
        '''
        Perform a linear decomposition of PCA components with the ghosts
        ''' 
        self.separator = SourceSeparator()
        self.separator.PerformTransform(self.loader.fftmag, len(self.loader.fftmag))

        results = np.zeros((self.ghost_basis_images.shape[0], self.separator.PCA_images.shape[0]))
        norms = np.zeros(self.ghost_basis_images.shape[0])
        for i, basis in enumerate(self.ghost_basis_images):
            for j, image in enumerate(self.separator.PCA_images):
                    results[i, j] = np.dot(image.reshape(-1), basis.reshape(-1)) / np.dot(basis.reshape(-1), basis.reshape(-1))
                    norms[i] = np.dot(basis.reshape(-1), basis.reshape(-1))
        
        print(np.square(results))
        np.set_printoptions(threshold=sys.maxsize)
        print(np.argmax(np.square(results), 0))
        #print(np.argmax(np.square(results), 1))
        #sums = np.sum(np.square(results), 1)
        #print(sums)
      
    def ConstructGramMatrix(self, images):
        # Matrix that checks if the set of images are linearly independant
        vectors = images.reshape(images.shape[0], -1)
        num_vectors = len(vectors)
        self.gram = np.zeros((num_vectors, num_vectors))
        for i, vector_i in enumerate(vectors):
            for j, vector_j in enumerate(vectors):
                self.gram[i, j] = np.dot(vector_i,  vector_j)
                
        print(f"The Gram Matrix has a determinate of {linalg.det(self.gram)}")   

    def DecomposeImages(self, hermite_order, images_to_decompose, results_directory):    
        hermite_constructor = HermiteConstructor()
        # Define the dimensions of the mask (e.g., 8x8)
        width, height = 1000, 1000
        rotation_angle_degrees = 45  # Change the angle as desired
        sizes = [5]
        hermite_constructor.CreateHermiteOfMaxOrder(width, height, hermite_order, rotation_angle_degrees, sizes)
        
        if(not os.path.exists(results_directory)):
            os.makedirs(results_directory)
            
        if not os.path.isfile(results_directory + '/coefficients_' + str(sizes[0]) +'.npy'):
            coefficients = np.zeros((images_to_decompose.shape[0], hermite_constructor.hermite_functions.shape[0]))
            hermite_norms = np.zeros(hermite_constructor.hermite_functions.shape[0])
            image_norms = np.zeros(images_to_decompose.shape[0])
            print(f"Projecting onto {hermite_constructor.hermite_functions.shape[0]} Hermite Functions")
            
            
            
            for j, hermite in enumerate(hermite_constructor.hermite_functions):
                size = sizes[j % len(sizes)]
                x = np.linspace(-width * (1/size), width * (1/size), width)
                y = np.linspace(-height * (1/size), height * (1/size), height)
                
                print(f"Projecting onto Hermite {j}")
                norm = simps(simps(hermite**2, y), x)
                for i, image in enumerate(images_to_decompose):
                        inner_product = simps(simps(hermite[484 : 516, 484 : 516] * np.real(image), y[484 : 516] ), x[484 : 516])
                        coefficients[i, j] = inner_product / norm

                hermite_norms[j] = norm
            
            for i, image in enumerate(images_to_decompose):
                image_norms[i] = simps(simps(np.real(image)**2, y[484 : 516] ), x[484 : 516])
            np.save(results_directory + '/coefficients_' + str(sizes[0]) +'.npy', coefficients)
        else:
            print("Loading Coefficients")
            coefficients = np.load(results_directory + '/coefficients_' + str(sizes[0]) +'.npy')

        # np.set_printoptions(threshold=sys.maxsize)
        #print(np.argmax(np.square(results), 1))

        # print(np.sum(np.square(results) * hermite_norms, axis = 1))
        # print(ghost_norms)
        
        # compression = np.sum(np.square(coefficients) * hermite_norms, axis = 1) / ghost_norms
        # print(np.sum(np.square(coefficients) * hermite_norms, axis = 1) / ghost_norms)
        
        # compression = np.sum(np.square(coefficients), axis = 0)
        # args = np.flip(np.argsort(compression))
        
        # plt.plot(np.flip(np.sort(compression)))
        # plt.show()

        recon_error = np.zeros((images_to_decompose.shape[0]))
        args = self.EnergyHermiteCoefficients(coefficients)
        
        reconstructed_images = np.zeros((images_to_decompose.shape[0], 32, 32))
        for i, image in enumerate(reconstructed_images):
            #print(i)
            for k, j in enumerate(args): #enumerate(hermite_constructor.hermite_functions):
                image += hermite_constructor.hermite_functions[j, 484 : 516, 484 : 516] * coefficients[i, j]
            error = np.abs((image - np.real(images_to_decompose[i]))) / (32 * 32)
            recon_error[i] = np.sum(np.sum(error))
        

        print(recon_error)

        self.SaveAllImages(reconstructed_images, 'reconstructed', results_directory)
        self.SaveAllImages(np.real(images_to_decompose), 'original', results_directory)
        self.SaveAllImages(hermite_constructor.hermite_functions[:, 484 : 516, 484 : 516], 'hermite_functions', results_directory)

        plt.plot(recon_error, marker='o')
        plt.xlabel('Basis Vectors')
        plt.ylabel('Cumulative Power')
        plt.title('Power Energy Analysis')
        plt.show()
        

    def EnergyHermiteCoefficients(self, coefficient_matrix):
        variance_per_basis = np.var(coefficient_matrix, axis=0)
        power_per_basis = np.sum(np.square(np.abs(coefficient_matrix)), axis = 0)
        args_var = np.flip(np.argsort(variance_per_basis))
        args_power = np.flip(np.argsort(power_per_basis))
        
        cumulative_variance = np.cumsum(variance_per_basis[args_var]) / np.sum(variance_per_basis)
        cumulative_power = np.cumsum(power_per_basis[args_power]) / np.sum(power_per_basis)
        plt.plot(cumulative_variance, marker='o')
        plt.xlabel('Basis Vectors')
        plt.ylabel('Cumulative Variance')
        plt.title('Variance Energy Analysis')
        plt.show()

        plt.plot(cumulative_power, marker='o')
        plt.xlabel('Basis Vectors')
        plt.ylabel('Cumulative Power')
        plt.title('Power Energy Analysis')
        plt.show()


        for i in range(len(args_power)):
            if cumulative_power[i] > 0.95:
                break
        print(i)
        return args_power[:i]
            
if __name__ == '__main__':
    tar = 'cifar-10-binary.tar.gz'
    files = ['cifar-10-batches-bin/data_batch_1.bin',
             'cifar-10-batches-bin/data_batch_2.bin',
             'cifar-10-batches-bin/data_batch_3.bin',
             'cifar-10-batches-bin/data_batch_4.bin',
             'cifar-10-batches-bin/data_batch_5.bin',
             'cifar-10-batches-bin/test_batch.bin']
    
    
    ghost_transform = GhostTransform(32)
    #ghost_transform.LoadRGBImagesFromTarfile(tar, files, (32, 32, 3), 10000)
    #ghost_transform.InitaliseGhosts(size_grid = 3, num_octants = 4, max_occurances = 4)
    ghost_transform.InitialiseRandomBigGhosts(size_grid = 3, num_octants = 4, num_ghosts = 50000, num_elements = 20)
    #print(ghost_transform.loader.gray_flattened.shape)
    
    #ghost_transform.HouseHolderQRDecomposition()
    ghost_transform.PCAOfGhosts()

    images_to_display = np.arange(0, 64, 1, dtype=int)
    # # images_to_display = np.arange(0, 1024, 16, dtype=int)
    
    
    #location = './ghost_transform_3_basis_4repeats_pca'
    location = './ghost_transform_3_big_pca'
    ghost_transform.SaveAllImages(ghost_transform.constructor.ghost_images, 'original_ghosts_', location)
    ghost_transform.SaveAllImages(np.abs(ghost_transform.constructor.receptive_field_images), 'original_fourierghosts_abs_', location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.constructor.receptive_field_images), 'original_fourierghosts_real_', location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.constructor.receptive_field_images), 'original_fourierghosts_imag_', location)
    
    ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images), 'ghosts_abs_', location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.ghost_basis_images), 'ghosts_real_', location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.ghost_basis_images), 'ghosts_angle_', location)

    ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images_ifft), "ghosts_ifft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.ghost_basis_images_ifft), "ghosts_ifft_angle_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.ghost_basis_images_ifft), "ghosts_ifft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.ghost_basis_images_ifft), "ghosts_ifft_imag_", location)
    
    ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images_fft), "ghosts_fft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.ghost_basis_images_fft), "ghosts_fft_angle_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.ghost_basis_images_fft), "ghosts_fft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.ghost_basis_images_fft), "ghosts_fft_imag_", location)
    
    ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images_noshift_ifft), "ghosts_noshift_ifft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.ghost_basis_images_noshift_ifft), "ghosts_noshift_ifft_angle_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.ghost_basis_images_noshift_ifft), "ghosts_noshift_ifft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.ghost_basis_images_noshift_ifft), "ghosts_noshift_ifft_imag_", location)

    ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images_1dshift_ifft), "ghosts_1dshift_ifft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.ghost_basis_images_1dshift_ifft), "ghosts_1dshift_ifft_angle_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.ghost_basis_images_1dshift_ifft), "ghosts_1dshift_ifft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.ghost_basis_images_1dshift_ifft), "ghosts_1dshift_ifft_imag_", location)


        
    ghost_transform.SaveAllImages(np.abs(ghost_transform.fourierghost_basis_images), "rf_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.fourierghost_basis_images), "rf_phase_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images), "rf_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images), "rf_imag_", location)

    ghost_transform.SaveAllImages(np.abs(ghost_transform.fourierghost_basis_images_fft), "rf_fft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.fourierghost_basis_images_fft), "rf_fft_phase_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images_fft), "rf_fft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images_fft), "rf_fft_imag_", location)

    ghost_transform.SaveAllImages(np.abs(ghost_transform.fourierghost_basis_images_ifft), "rf_ifft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.fourierghost_basis_images_ifft), "rf_ifft_phase_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images_ifft), "rf_ifft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images_ifft), "rf_ifft_imag_", location)

    ghost_transform.SaveAllImages(np.abs(ghost_transform.fourierghost_basis_images_noshift_fft), "rf_noshift_fft_abs_", location)
    ghost_transform.SaveAllImages(np.angle(ghost_transform.fourierghost_basis_images_noshift_fft), "rf_noshift_fft_phase_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images_noshift_fft), "rf_noshift_fft_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images_noshift_fft), "rf_noshift_fft_imag_", location)


    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images_shiftedrow), "rf_shiftedrow_imag_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images_shiftedrow), "rf_shiftedrow_real_", location)
    ghost_transform.SaveAllImages(np.imag(ghost_transform.fourierghost_basis_images_shiftedcol), "rf_shiftedcol_imag_", location)
    ghost_transform.SaveAllImages(np.real(ghost_transform.fourierghost_basis_images_shiftedcol), "rf_shiftedcol_real_", location)


    # ghost_transform.QRWithRadonFlattening()
    # ghost_transform.SaveAllImages(ghost_transform.ghost_sinograms, "sinograms_", location)
    # ghost_transform.SaveAllImages(np.abs(ghost_transform.fourier_slices), "fourierslices_", location)
    # ghost_transform.SaveAllImages(ghost_transform.ghost_basis_images, "ghosts_radon_", location)
    # ghost_transform.SaveAllImages(np.abs(ghost_transform.ghost_basis_images), "ghosts_abs_radon_", location)


    # hermite_order = 40
    # ghost_transform.DecomposeGhosts(10, ghost_transform.constructor.ghost_images)
    # ghost_transform.DecomposeImages(hermite_order, ghost_transform.constructor.receptive_field_images, './rf_recon_' + 'order_' + str(hermite_order))

    #ghost_transform.ConstructGramMatrix(ghost_transform.basis_images)
    #ghost_transform.LinearDecompositionGhosts()
    

    