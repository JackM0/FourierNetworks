import tarfile
import numpy as np

class DataLoader:
    """
    Class with functions to load different datasets and perform basic preprocessing
    """
    def init(self):
        """
        DataLoader Constructor
        """
        self.dataset = np.array([])
        self.images = np.array([[[]]])
        self.h = 0
        self.w = 0
        self.num_images = 0
        
    #   @fn FourierTransformImages (0)
    def FourierTransformImages(self):
        """
        Compute the FFT of the Grayscale Images
        """
        fft = np.fft.fftshift(np.fft.fft2(self.gray))
        self.fftmag = np.abs(fft)
        self.fftphase = np.angle(fft)

    #   @fn LoadTarfile (0)
    def LoadTarfile(self, tar, files, im_shape, images_per_file):
        """
        Load in a dataset from a Tarfile
        @param tar (string) : The name of the .gz file
        @param files list(string) : A list of the different bin names in the tarfile
        @param im_shape (int, int, int) : The shape of the images
        @param images_per_file (int) : Images per file/bin
        """
        self.h = im_shape[0]
        self.w = im_shape[1]
        num_channels = im_shape[2]
        num_files = len(files)
        
        self.num_images = num_files * images_per_file
        
        with tarfile.open(tar) as tar_object:
            # E.g Each file contains 10,000 color images and 10,000 labels
            fsize = images_per_file * (self.h * self.w * num_channels) + images_per_file

            # E.g There are 6 files (5 train and 1 test), so num_files = 6
            buffr = np.zeros(fsize * num_files, dtype='uint8')

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
            num_bytes = self.h * self.w * num_channels + 1
            self.interger_labels = buffr[::num_bytes]
            
            # Pixels are everything remaining after we delete the labels
            pixels = np.delete(buffr, np.arange(0, buffr.size, num_bytes))
            self.dataset = pixels.reshape(-1, num_bytes - 1).astype('float32') / 255

    #   @fn TripleChannelUnflatten (0)
    def TripleChannelUnflatten(self):
        """
        Unpack a 3-Channel Image Dataset into its individual channels and compute a single channel gray scale
        """
        pixels_per_image = self.h * self.w
        channel_1 = self.dataset[:, 0 : pixels_per_image].reshape(self.num_images, self.h, self.w)
        channel_2 = self.dataset[:, pixels_per_image : 2 * pixels_per_image].reshape(self.num_images, self.h, self.w)
        channel_3 = self.dataset[:, 2 * pixels_per_image : 3 * pixels_per_image].reshape(self.num_images, self.h, self.w)
        
        self.images = np.zeros((self.num_images, self.h, self.w, 3))
        self.images[:, :, :, 0] = channel_1
        self.images[:, :, :, 1] = channel_2
        self.images[:, :, :, 2] = channel_3
        self.rgb2gray(channel_1, channel_2, channel_3)
        
        self.gray_flattened = self.gray.reshape((self.num_images, self.h * self.w))
    
    #   @fn rgb2gray (1)
    def rgb2gray(self, r, g, b):
        """
        Convert 3-channels of a set of images into grayscale
        @param r np.array([[[]]]) : Array of Red Images
        @param g np.array([[[]]]) : Array of Green Images
        @param b np.array([[[]]]) : Array of Blue Images
        """
        self.gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
