import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import hermitenorm
from scipy.special import hermite
from scipy.integrate import simps

import os

class HermiteConstructor:
    """
    Class with functions to create 2D Hermite Functions
    """
    def init(self):
        return
    
    def CreateHermiteOfMaxOrder(self, width, height, max_order, rotation_angle_degrees, sizes):
        # Define the dimensions of the mask (e.g., 8x8)
        width, height = 1000, 1000
        # Specify the order and rotation angle of the Hermite functions
        max_order = max_order  # The maximum order number

        orders = []

        for i in range(max_order + 1):
            for j in range(max_order + 1):
                orders.append((i, j))
        orders =  sorted(orders, key=lambda pair: pair[0] + pair[1])
        rotation_angle_degrees = 45  # Change the angle as desired
        
        self.CreateHermiteSet(width, height, orders, rotation_angle_degrees, sizes)
    
    def CreateHermiteSet(self, width, height, orders, rotation_angle_degrees, sizes):  
        self.width = width
        self.height = height
        self.num_orders = len(orders)
        self.num_sizes = len(sizes)
        self.num_functions = self.num_orders * self.num_sizes
        self.hermite_functions = np.zeros((self.num_functions, width, height))
        
        hermite_array_path = './hermite_arrays/'
        print(f"Creating/Loading {self.num_functions} Hermite Functions")
        for i in range(self.num_orders):
            n_order = orders[i][0]
            m_order = orders[i][1]
            for j in range(self.num_sizes):
                print(f"Creating Hermite {j + i * self.num_sizes}")
                size = sizes[j]
                hermite_filename = 'hermite_width' + str(width) + 'height_' + str(height) + 'n_' + str(n_order) + 'm_' + str(m_order) + 'angle_' + str(rotation_angle_degrees) + 'size_' + str(size) + '.npy'
                if not os.path.isfile(hermite_array_path + hermite_filename):
                    self.hermite_functions[j + i * self.num_sizes, :, :] = self.CreateCheckerHermite(width, height, n_order, m_order, rotation_angle_degrees, size)   
                    np.save(hermite_array_path + hermite_filename, self.hermite_functions[j + i * self.num_sizes, :, :])
                else:
                    self.hermite_functions[j + i * self.num_sizes, :, :] = np.load(hermite_array_path + hermite_filename)
                #self.hermite_functions[j + i * self.num_sizes, :, :] = self.CreateCheckerHermite(width, height, n_order, m_order, rotation_angle_degrees, size) 
        
        # x = np.linspace(-width / 8, width / 8, width)
        # y = np.linspace(-height / 8, height / 8, height)
        
        # for hermite in self.hermite_functions:
        #     for hermite2 in self.hermite_functions:
        #         integral = np.abs(simps(simps(hermite * hermite2 , y), x))
        #         if np.isclose(integral, 1.0, atol=1e-3) or np.isclose(integral, 0, atol=1e-3):
        #             pass
        #         else:
        #             print(integral)
    
    def CreateCheckerHermite(self, width, height, n_order, m_order, rotation_angle_degrees, size):
        # Create the alternating +1 and -1 mask
        mask = np.ones((width, height), dtype=int)
        mask[1::2, ::2] = -1
        mask[::2, 1::2] = -1

        # Define the grid of the Hermite function
        x = np.linspace(-width * (1/size), width * (1/size), width)
        y = np.linspace(-height * (1/size), height * (1/size), height)
        X, Y = np.meshgrid(x, y)

        # Generate the rotated 2D Hermite function
        hermite_function = self.Hermite2D(X, Y, n_order, m_order, rotation_angle_degrees)
        checker_hermite = hermite_function #*mask
        #print(simps(simps(checker_hermite**2, y), x))
        return checker_hermite
    
    def Hermite2D(self, x, y, n, m, angle_degrees=0):
        # Hn = self.HermitePolynomial(n, x)
        # Hm = self.HermitePolynomial(m, y)

        # Rotate the Hermite function using scipy's rotate function
        hermite_function = (hermite(n)(x) * hermite(m)(y)) * np.exp(-0.5 * (x**2 + y**2)) / (2**((n+m)/2) * np.sqrt(np.math.factorial(n) * np.math.factorial(m) * np.pi))
        rotated_hermite_function = rotate(hermite_function, angle_degrees, reshape=False)

        return rotated_hermite_function

    def HermitePolynomial(self, n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return 2.0 * x
        else:
            return 2.0 * x * self.HermitePolynomial(n - 1, x) - 2.0 * (n - 1) * self.HermitePolynomial(n - 2, x)
        
    def DisplayAllFunctions(self):
        for i in range(2):
            images_to_display = np.arange(0 + i * 64, (i + 1) * 64, 1, dtype=int)
            num_images = len(images_to_display)
            display_rows_cols = int(num_images**0.5)

            fig=plt.figure(figsize=(32, 32))          
            for j, image_index in enumerate(images_to_display):
                ax=fig.add_subplot(display_rows_cols, display_rows_cols, j+1)
                ax.imshow(self.hermite_functions[image_index, 484 : 517, 484 : 517], cmap=plt.cm.bone)

            plt.show()


if __name__ == '__main__':
    hermite_constructor = HermiteConstructor()

    # Define the dimensions of the mask (e.g., 8x8)
    width, height = 1000, 1000
    # Specify the order and rotation angle of the Hermite functions
    max_order = 20  # The maximum order number
    size = [5]
    orders = []

    for i in range(max_order + 1):
        for j in range(max_order + 1):
            orders.append((i, j))
            
    orders =  sorted(orders, key=lambda pair: pair[0] + pair[1])
    rotation_angle_degrees = 45  # Change the angle as desired
    print(orders)
    hermite_constructor.CreateHermiteSet(width, height, orders, rotation_angle_degrees, size)
    hermite_constructor.DisplayAllFunctions()
    

