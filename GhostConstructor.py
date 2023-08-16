import numpy as np
from radon.farey import Farey
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
        
    def PossibleConvolutionGhosts(self, size_grid):
        """
        Create Ghosts using all possible combinations of Ghost Kernels for a given grid size / Order of Farey Vectors
        @param size_grid (int) : Specify the size_grid from which we want the 2PSE to have to fit within, also the order of Farey vectors that the gradient between pixels must be an element of
        """
        self.Generate2PSEs(size_grid)
        num_elements = self.structuring_elements.shape[0]
        self.ghosts = []
        
        for i in range(2 ** num_elements):
            self.ghosts.append(Ghosts(256))    
            
        for i, ghost in enumerate(self.ghosts):
            kernel_ids = []
            b_string = f'{bin(i)[2:].zfill(num_elements)}'
            print(b_string)
            for k, bit in enumerate(b_string):
                if bit == '1':
                    kernel_ids.append(k)
            #print(kernel_ids)
            self.CreateConvolutionGhost(ghost, kernel_ids)
    
    def Generate2PSEs(self, size_grid):
        farey_vector_generator = Farey()
        farey_vector_generator.generate(size_grid - 1, 2)
        vectors = np.array(farey_vector_generator.vectors)
        
        self.structuring_elements = np.empty((vectors.size, 2)).astype(int)
        for i, vector in enumerate(vectors):
            self.structuring_elements[i] = self.FareyTo2PSE(vector)
  
        return self.structuring_elements
    
    def FareyTo2PSE(self, farey_vector):
        return np.array([np.imag(farey_vector), np.real(farey_vector)])
        
    def CreateConvolutionGhost(self, ghost, kernel_ids):
        for id in kernel_ids:
            ghost.ConvolveWithGhostKernel(self.structuring_elements[id])
            
    def PlotAllGhosts(self):
        for ghost in self.ghosts:
            ghost.PlotGhost()
        
if __name__ == "__main__":
    constructor = GhostCreator(256)
    constructor.PossibleConvolutionGhosts(3)
    constructor.PlotAllGhosts()