import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt


class Kohonen:
    def __init__(self, M: int, K: int) -> None:
        """
        Initial function

        Args:
            M (int): size of input layer
            K (int): size of layer
        """
        self.M = M
        self.K = K
        self.W = None

    def generate_W(self) -> None:
        """
        Generate the weight matrix for Kohonen's self organizing map.

        The weight matrix is initialized with random values between 0.5 - 1/sqrt(M) and 0.5 + 1/sqrt(M)
        where M is the size of the input layer.
        """
        self.W = np.random.uniform(0.5 - 1 / (self.M ** 0.5), 0.5 + 1 / (self.M ** 0.5) + 1, size=(self.K, self.M))
    
    @staticmethod
    def _eta_function(n, eta_0=0.1, t2=2000):
        return eta_0 * np.exp(-n / t2)
    
    @staticmethod
    def plot_eta():
        x = np.linspace(0, 10**4, 1000)
        plt.plot(x, Kohonen._eta_function(x))
        plt.show()
    
    @staticmethod
    def _sigma_function(n, sig_0=10, t1=1000):
        return (sig_0 * np.exp(-n / t1))
    
    @staticmethod
    def plot_sigma():
        x = np.linspace(0, 10**4, 1000)
        plt.plot(x, Kohonen._sigma_function(x))
        plt.show()
    
    def _iteration(self, in_x, n):
        winner_index = np.argmax([distance.euclidean(in_x, i) for i in self.W])
        size_of_side = int(self.K ** 0.5)
        winner = (winner_index // size_of_side, winner_index % size_of_side)
        
        distance_Gauss = np.zeros((size_of_side, size_of_side))
        for i in range(size_of_side):
            for j in range(size_of_side):
                distance_Gauss[i][j] = np.exp(-distance.euclidean(winner, (i, j)) / (2 * self._sigma_function(n)))
                
        dW = np.zeros((self.K, self.M))
        for i in range(size_of_side):
            for j in range(size_of_side):
                dW[i * size_of_side + j] = Kohonen._eta_function(n) * distance_Gauss[i][j] * (in_x - self.W[i * size_of_side + j])

        self.W = self.W + dW
            

def main():
    kh = Kohonen(7, 4)
    # Kohonen.plot_eta()
    # Kohonen.plot_sigma()
    kh.generate_W()
    kh._iteration(np.random.uniform(0, 1, size=(7)), 10000)


if __name__ == '__main__':
    main()