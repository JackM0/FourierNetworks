import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

im_path = "./images/"

class Ghosts:
    """
    Create Ghosts from Farey Vectors, manipulate them, calculate their Receptive Fields and Display them
    """

    def __init__(self, N):
        """
        Ghost Constructor
        @param N (int) : The Image Dimension
        """
        self.N = N
        self.ghost = np.zeros((N , N))
        self.cropped_ghost = np.array([[]])
        self.fttghost = np.array([[]])
        self.name = ""
        self.fail = False

    #   @fn PlotGhost (0)   
    def PlotGhost(self, location):
        """
        Plots the Ghost and the Magnitude and Phase of its Fourier Transform
        """
        f, axs = plt.subplots(1,3)
        axs[1].imshow(np.abs(self.fttghost))
        axs[1].title.set_text("Ghost FFT")
        axs[2].imshow(np.angle(self.fttghost))
        axs[2].title.set_text("FFT Phase")
        axs[0].imshow(self.cropped_ghost)
        axs[0].title.set_text("Ghost")
        
        f.suptitle(self.name)
        f.set_figheight(10)
        f.set_figwidth(10)
        f.savefig('./' + location + '/' + self.name + '.png')
        plt.close(f)
        plt.show()

    #   @fn ConvolveWithGhostKernel (0)
    def ConvolveWithGhostKernel(self, pixel):
        """
        Convolves the current ghost/set of kernels with a new ghost kernel specified by a (2PSE) Relative Pixel Position Vector
        @param pixel_position ([int, int]) : The relative position of the -1 pixel in the kernel from the +1 pixel
        """
        # Create the Kernel and Convolved with the current Ghost
        kernel = self.CreateGhostKernel(pixel)
        self.cropped_ghost = signal.convolve(self.cropped_ghost, kernel) if self.cropped_ghost.size > 0 else kernel
        
        # Adjust the Ghosts name to indicate which kernel has been applied to it
        self.name += np.array2string(pixel)

        # Embbed the resultant Ghost in an Image specified by the image size
        try:
            self.ghost[int(self.N / 2) : int(self.cropped_ghost.shape[0] + self.N / 2), int(self.N / 2) : int(self.cropped_ghost.shape[1] + self.N / 2)] = self.cropped_ghost
        except:
            self.fail = True
        self.FFT_Ghost()
    
    #   @fn FareyVectorsToCoords (0)
    def FareyVectorsToCoords(self, vectors):
        """
        Converts a List of Farey Vectors into (x,y) pixel coordinates
        @param vectors ([complex numbers]) : A list of complex numbers
        """
        vectors = np.array(vectors)
        
        pixels = np.empty((vectors.size, 2))
        for i, vector in enumerate(vectors):
            pixels[i] = self.FareyToPixelCoord(vector)
        
        return pixels
                
    #   @fn BuildGhostFromPixels (0)
    def BuildGhostFromPixels(self, pixels, build_fft = 1, rebuild = 1):
        """
        Builds/Loads the List of Pixel Coordinates into an actual Image/2D Array
        @param rebuild (int) : Indicate whether the ghost should be completely cleared before adding in the new pixels
        @param build_fft (int) : Indicate whether the fft of the ghost should also be calculated now
        """
        if rebuild:
            self.ghost = np.zeros((self.N, self.N))
            
        for pixel in pixels:
            self.ghost[int(pixel[0]), int(pixel[1])] = 1
        
        if build_fft:
            self.FFT_Ghost()
    
    #   @fn CopyGhost (0)
    def CopyGhost(self, initial_ghost):
        """
        Create a Copy of an exhisting Ghost Instance
        @param initial_ghost (Ghost()) : An exhisting ghost instance that will be copied over to this one
        """
        self.N = initial_ghost.N
        self.ghost = initial_ghost.ghost
        self.fttghost = initial_ghost.fftghost
        self.convoledkernels = initial_ghost.convoledkernels
    
    #   @fn CombineGhosts (0)
    def CombineGhosts(self, additional_ghost):
        """
        Load the Pixels from an already exhisting ghost into this one, then rebuilds the ghost
        @param additional_ghost (Ghost()) : An exhisting ghost instance that will be combined with this one
        @returns 1 for success, -1 for failure
        """
        if (self.N != additional_ghost.N):
            print("Error: Ghosts are Different Sizes")
            return -1
        
        self.ghost = self.ghost + additional_ghost.ghost
        self.FFT_Ghost()
        return 0
    
    #   @fn ShiftGhost (0)    
    def ShiftGhost(self, translation):
        """
        Shift the ghost by the amount given by the translation vector and rebuild the ghost
        @param translation ([x,y]) : Vector to translate Ghost by (Reminded that y > 0 moves the ghost towards the bottom of the image)
        """
        self.ghost = np.roll(self.ghost, translation[0], axis = 0)
        self.ghost = np.roll(self.ghost, translation[1], axis = 1)
        self.FFT_Ghost()
    
    #   @fn RotateGhost (0)  
    def RotateGhost(self, angle):
        """
        Rotate the ghost by some angle
        @param angle (float) : Angle to rotate the ghost by counter-clockwise
        """
        ndimage.rotate(self.ghost, angle, order = 0)
        self.FFT_Ghost()
            
    #   @fn FFT_Ghost (1)        
    def FFT_Ghost(self):
        """
        Calculates the 2D Inverse FFT of the Ghost
        """
        self.fttghost = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.ghost)))

    #   @fn CreateGhostKernel (1)
    def CreateGhostKernel(self, pixel_position):
        """
        Create Ghost Kernel (2PSE) from a Relative Pixel Position Vector
        @param pixel_position ([int, int]) : The relative position of the -1 pixel in the kernel from the +1 pixel
        """
        kernel = np.zeros((abs(pixel_position[0]) + 1, abs(pixel_position[1]) + 1))

        if (pixel_position[1] < 0):
            kernel[0, -1] = 1
            kernel[pixel_position[0], pixel_position[1] - 1] = -1
        else:
            kernel[0, 0] = 1
            kernel[pixel_position[0], pixel_position[1]] = -1
        
        return kernel

    #   @fn FareyToPixelCoord (1) 
    def FareyToPixelCoord(self, farey_vector):
        """
        Convert a Farey Vector (Complex Number) into a pixel coordinate
        @param farey_vector (complex number) : A Farey Vector
        @returns [x,y] Pixel Coordinate
        """
        return [-np.imag(farey_vector) + self.N / 2, np.real(farey_vector) + self.N / 2, ]


if __name__ == "__main__":
    # test_ghost = Ghosts(256)
    # farey_vectors = Farey()
    # farey_vectors.generate2(3, 8)

    # test_ghost.FareyVectorsToCoords(farey_vectors.vectors)
    # test_ghost.BuildGhost()

    # test_ghost.PlotGhost()
    # test_ghost.ShiftGhost([3,3])
    # test_ghost.PlotGhost()
    test_ghost1 = Ghosts(256)
    pses = test_ghost1.Generate2PSEs(3)
    test_ghost1.ConvolveWithGhostKernel(pses[0])
    test_ghost1.PlotGhost()