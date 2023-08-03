import numpy as np
from radon.farey import Farey
import matplotlib.pyplot as plt

class Ghosts:

    def __init__(self, N):
        self.N = N
        self.ghost = np.zeros((N , N))

    
    def FareyVectorsToCoords(self, vectors):
        coords = np.empty((vectors.size, 2))
        for i, vector in enumerate(vectors):
            coords[i] = [np.real(vector) + self.N / 2, - np.imag(vector) + self.N / 2]
        return coords       

    def SetPixels(self, pixel_list):
        for pixel in pixel_list:
            self.ghost[int(pixel[0]), int(pixel[1])] = 1



if __name__ == "__main__":
    ghost = Ghosts(256)
    farey_vectors = Farey()
    farey_vectors.generate2(7, 8)
    print("hello")
    print(farey_vectors.vectors)
    pixels = ghost.FareyVectorsToCoords(np.array(farey_vectors.vectors))
    ghost.SetPixels(pixels)
    print(ghost.ghost)
    
    plt.imshow(np.abs(np.fft.ifftshift(np.fft.ifft(np.fft.fftshift(ghost.ghost)))))
    plt.show()

        
