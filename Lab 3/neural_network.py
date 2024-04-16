import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

rand = np.random.RandomState(0)

class Kohonen:
    def __init__(self, M: int, K: int):
        """
        Initial function

        Args:
            M (int): size of input layer
            K (int): size of side of square
        """
        self.M = M
        self.K = K
        self.W = None

    def generate_W(self):
        self.W = rand.uniform(0, 1, (self.K, self.K, self.M)).astype(float)

    def _find_winner(self, in_x):
        distSq = (np.square(self.W - in_x)).sum(axis=2)
        return np.unravel_index(np.argmin(distSq, axis=None), distSq.shape)

    def _iteration(self, in_x, learn_rate, radius_sq, step=3):
        g, h = self._find_winner(in_x)
        if radius_sq < 1e-3:
            self.W[g, h, :] += learn_rate * (in_x - self.W[g, h, :])
            return
        for i in range(max(0, g - step), min(self.W.shape[0], g + step)):
            for j in range(max(0, h - step), min(self.W.shape[1], h + step)):

                dist_sq = np.square(i - g) + np.square(j - h)
                dist_func = np.exp(-dist_sq / 2 / radius_sq)
                self.W[i, j, :] += learn_rate * dist_func * (in_x - self.W[i, j, :])

    def calculate(self, dataset, count_iter,
                  learn_rate_0=.1, lr_decay=.1,
                  radius_0=1, radius_decay=.1):
        dataset_copy = dataset.copy()
        for i in range(count_iter):
            learn_rate = learn_rate_0 * np.exp(-i * lr_decay)
            radius_sq = radius_0 * np.exp(-i * radius_decay)
            rand.shuffle(dataset_copy)
            if i % 1 == 0:
                print(round(i / count_iter * 100, 1))
            for data in dataset_copy:
                self._iteration(data, learn_rate, radius_sq)


def main():
    kh = Kohonen(2, 10)
    kh.generate_W()
    train_data = np.append(np.random.uniform(0, 1, (5000, 2)), np.random.uniform(0, 1, (5000, 2)), axis=0)
    plt.scatter(*zip(*train_data))
    plt.show()
    plt.scatter(*zip(*kh.W.reshape(1, 100, 2)[0]))
    plt.show()
    kh.calculate(train_data, 10)
    plt.scatter(*zip(*kh.W.reshape(1, 100, 2)[0]))
    plt.show()


if __name__ == '__main__':
    main()
