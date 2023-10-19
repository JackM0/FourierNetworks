import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from scipy.special import hermitenorm

class HermiteConstructor:
    """
    Class with functions to create 2D Hermite Functions
    """
    def init(self):
        return
    
    def CreateHermiteSet(self, width, height, orders, rotation_angle_degrees):
        self.width = width
        self.height = height
        self.num_functions = len(orders)
        self.hermite_functions = np.zeros((self.num_functions, width, height))

        for i in range(self.num_functions):
            n_order = orders[i][0]
            m_order = orders[i][1]
            self.hermite_functions[i, :, :] = self.CreateCheckerHermite(width, height, n_order, m_order, rotation_angle_degrees)
            
    
    def CreateCheckerHermite(self, width, height, n_order, m_order, rotation_angle_degrees):
        # Create the alternating +1 and -1 mask
        mask = np.ones((width, height), dtype=int)
        mask[1::2, ::2] = -1
        mask[::2, 1::2] = -1

        # Define the grid of the Hermite function
        x = np.linspace(-width / 8, width / 8, width)
        y = np.linspace(-height / 8, height / 8, height)
        X, Y = np.meshgrid(x, y)

        # Generate the rotated 2D Hermite function
        hermite_function = self.Hermite2D(X, Y, n_order, m_order, rotation_angle_degrees)
        
        return np.abs(hermite_function) * mask
    
    def Hermite2D(self, x, y, n, m, angle_degrees=0):
        Hn = self.HermitePolynomial(n, x)
        Hm = self.HermitePolynomial(m, y)

        # Rotate the Hermite function using scipy's rotate function
        hermite_function = Hn * Hm * np.exp(-x**2 / 2) * np.exp(-y**2 / 2)
        rotated_hermite_function = rotate(hermite_function, angle_degrees, reshape=False)
        Hn = hermitenorm(n)(x)
        Hm = hermitenorm(m)(y)
        
        print(np.sum(hermite_function * hermite_function * np.exp(-x**2) * np.exp(-y**2))) # TODO

        return rotated_hermite_function

    def HermitePolynomial(self, n, x):
        if n == 0:
            return 1.0
        elif n == 1:
            return 2.0 * x
        else:
            return 2.0 * x * self.HermitePolynomial(n - 1, x) - 2.0 * (n - 1) * self.HermitePolynomial(n - 2, x)
        
    def DisplayAllFunctions(self):
        fig=plt.figure(figsize=(self.width, self.height))        
        for i, image in enumerate(self.hermite_functions):
            ax=fig.add_subplot(int(self.num_functions / 5) + 1, 5, i + 1)
            ax.imshow(image, cmap=plt.cm.bone)

        plt.show()


if __name__ == '__main__':
    hermite_constructor = HermiteConstructor()

    # Define the dimensions of the mask (e.g., 8x8)
    width, height = 32, 32
    # Specify the order and rotation angle of the Hermite functions
    max_order = 5  # The maximum order number

    orders = []

    for i in range(max_order + 1):
        for j in range(max_order + 1):
            orders.append((i, j))
    orders = sorted(orders, key=lambda pair: max(pair))
    rotation_angle_degrees = 45  # Change the angle as desired
    print(len(orders))
    hermite_constructor.CreateHermiteSet(width, height, orders, rotation_angle_degrees)
    hermite_constructor.DisplayAllFunctions()

