import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy import signal

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
        axs[0].imshow(self.ghost)
        axs[0].title.set_text("Ghost")
        
        axs[0].set_xticks(np.arange(1,32, 2))
        axs[0].set_yticks(np.arange(1,32, 2))
        axs[1].set_xticks(np.arange(1,32, 2))
        axs[1].set_yticks(np.arange(1,32, 2))
        axs[2].set_xticks(np.arange(1,32, 2))
        axs[2].set_yticks(np.arange(1,32, 2))
        
        f.suptitle(self.name)
        f.set_figheight(10)
        f.set_figwidth(10)
        f.savefig('./' + location + '/' + self.name + '.png')
        plt.close(f)
        plt.show()

    #   @fn ConvolveWithGhost (0)
    def ConvolveWithGhost(self, kernel):
        """
        Convolves the current ghost/set of kernels with a new ghost kernel specified by a (2PSE) Relative Pixel Position Vector
        @param kernel ([int, int]) : The kernel to be convoled with the Ghost
        """
        # Create the Kernel and Convolved with the current Ghost
        self.cropped_ghost = signal.convolve(self.cropped_ghost, kernel) if self.cropped_ghost.size > 0 else kernel

        # Embbed the resultant Ghost in an Image specified by the image size
        if self.cropped_ghost.shape[0] > self.N or self.cropped_ghost.shape[1] > self.N:
            self.fail = True
        else:
            self.EmbebbedGhost()

        self.FFT_Ghost()
    
    #   @fn EmbebbedGhost (1)
    def EmbebbedGhost(self):
        h = self.cropped_ghost.shape[0]
        w = self.cropped_ghost.shape[1]
        
        a = (self.N - h) // 2
        aa = self.N - a - h

        b = (self.N - w) // 2
        bb = self.N - b - w
        
        self.ghost = np.pad(self.cropped_ghost, pad_width=((a, aa), (b, bb)), mode='constant')
        return

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
        self.fttghost = np.fft.ifft2(np.fft.ifftshift(self.ghost))