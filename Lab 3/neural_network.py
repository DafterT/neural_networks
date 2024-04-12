import numpy as np


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
        self.W = np.random.uniform(0.5 - 1 / (self.M ** 0.5), 0.5 + 1 / (self.M ** 0.5) + 1, size=(self.M, self.K))
        

def main():
    kh = Kohonen(20, 5)
    kh.generate_W()
    print(kh.W)

if __name__ == '__main__':
    main()