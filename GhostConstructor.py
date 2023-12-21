import numpy as np
from radon.farey import Farey
import os
import random

from Ghosts import Ghosts

class GhostCreator:
    """
    Class that will Create a Set of Ghosts using possible combinations of 2PSE
    """
    
    def __init__(self, N):
        """
        GhostCreator Constructor
        @param N (int) : The Image Dimension
        """
        self.N = N
        self.structuring_elements = np.array([[]])
        self.ghosts = np.array([[]])
    
    #   @fn PlotAllGhosts (0)    
    def PlotAllGhosts(self, location):
        if(not os.path.exists(location)):
            os.makedirs(location)

        for ghost in self.ghosts:
            ghost.PlotGhost(location)

    #   @fn CreateGhostsFrom2PSE (0)
    def CreateGhostsFrom2PSE(self, size_grid, num_octants, max_occurances):
        """
        Create Ghosts using all possible combinations of Ghost Kernels for a given grid size / Order of Farey Vectors
        @param size_grid (int) : Specify the size_grid from which we want the 2PSE to have to fit within, also the order of Farey vectors that the gradient between pixels must be an element of
        @param num_octants (int) : The number of Octants from which the Farey vectors should be picked from
        @param max_occurances (int) : The maximum number of times a single 2PSE should be used in any ghost it is used to create
        """
        # Generate the [p,q] for a set of 2PSE
        self.Generate2PSEs(size_grid, num_octants)
        num_elements = self.structuring_elements.shape[0]
        
        # Create set of Ghosts using all possible combinations of the 2PSE in self.structuring_elements
        self.ghosts = []
        for i in range((max_occurances + 1) ** num_elements):
            ghost = Ghosts(self.N)
            self.ghosts.append(ghost)
            
            kernel_ids = []
            s = ""
            n = i

            # Create a base-max_occurances number with len(self.structuring_elements) digits to specify which elements are used and how many times
            while n:
                s = str(n % (max_occurances + 1)) + s
                n //= (max_occurances + 1)
            s = s.zfill(num_elements)  
            print(s)
            
            for kernel_id, num_occurances in enumerate(s):
                for _ in range(int(num_occurances)):
                    kernel_ids.append(kernel_id)

            self.CreateConvolutionGhost(ghost, kernel_ids)
                
        self.ghosts = self.ghosts[1:]
        self.ghost_images = np.array([Ghost.ghost for Ghost in self.ghosts])
        self.receptive_field_images = np.array([Ghost.fttghost for Ghost in self.ghosts])
    
    def CreateBigGhosts(self, size_grid, num_octants, num_ghosts, num_elements_to_use):
         # Generate the [p,q] for a set of 2PSE
        self.Generate2PSEs(size_grid, num_octants)
        num_elements = self.structuring_elements.shape[0]
        
        self.ghosts = []
        for _ in range(num_ghosts):
            ghost = Ghosts(self.N)
            self.ghosts.append(ghost)
            kernel_ids = []
            for _ in range(num_elements_to_use):
                k_id = np.random.randint(0, num_elements)
                if k_id < num_elements:
                    kernel_ids.append(k_id)
            
            self.CreateConvolutionGhost(ghost, kernel_ids)

        self.ghost_images = np.array([Ghost.ghost for Ghost in self.ghosts])
        self.receptive_field_images = np.array([Ghost.fttghost for Ghost in self.ghosts])
        
        return
    
    #   @fn Generate2PSEs (1)
    def Generate2PSEs(self, size_grid, num_octants):
        """
        Create all the possible [p,q] identifiers for the set of 2PSE that fit in a given grid size and from Farey vectors from specific number of octants
        @param size_grid (int) : Specify the size_grid from which we want the 2PSE to have to fit within, also the order of Farey vectors that the gradient between pixels must be an element of
        @param num_octants (int) : The number of Octants from which the Farey vectors should be picked from
        """
        farey_vector_generator = Farey()
        farey_vector_generator.generate(size_grid - 1, num_octants)
        vectors = np.array(farey_vector_generator.vectors)
        vectors = np.unique(vectors)

        print(f"Unique Farey Vectors in a {size_grid} X {size_grid} Grid from {num_octants} Octants: {vectors}")
        
        self.structuring_elements = np.empty((vectors.size, 2)).astype(int)
        for i, vector in enumerate(vectors):
            self.structuring_elements[i] = self.FareyTo2PSE(vector)
        
        print(f"2PSE from this set of Farey Vectors: {self.structuring_elements}")
        
        return self.structuring_elements
    
    #   @fn CreateConvolutionGhost (1)
    def CreateConvolutionGhost(self, ghost, kernel_ids):
        """
        Create Ghost by convolving together many different 2PSE specified by their index in self.structuring_elements
        @param ghost Ghost() : The Ghost that will be build from the 2PSE
        @param kernel_ids list() : A list of Kernel Ids / Indexs specifying which 2PSE from self.structuring_elements will be used to create a ghost
        """
        for id in kernel_ids:
            kernel = self.CreateGhostKernel(self.structuring_elements[id])
            ghost.ConvolveWithGhost(kernel)  
            # Adjust the Ghosts name to indicate which kernel has been applied to it
            ghost.name += np.array2string(self.structuring_elements[id])
            
    #   @fn CreateGhostKernel (2)
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
    
    #   @fn FareyTo2PSE (2)
    def FareyTo2PSE(self, farey_vector):
        """
        Create a 2PSE identified written as [p,q] from a given Farey Vector
        @param farey_vector ([int, int]) : The Farey Vector used to specify the [p,q] for a 2PSE
        """
        return np.array([np.imag(farey_vector), np.real(farey_vector)])
        
if __name__ == "__main__":
    constructor = GhostCreator(32)
    constructor.CreateGhostsFrom2PSE(size_grid = 3, num_octants = 4, max_occurances = 1)
    constructor.PlotAllGhosts("./ghosts_3")