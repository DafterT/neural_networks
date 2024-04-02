import numpy as np
from PIL import Image

from constants import image_size


class Hopfild:

    def __init__(self, N, k):  # инициализация
        self.N = N  # количество нейронов
        self.K = k  # максимальное количество эпох распознавания сигнала
        self.W = np.zeros((self.N, self.N))  # матрица взаимодействий (весов)

    def remember(self, M):  # метод запоминания образов
        for X in M:
            self.W += X.dot(X.T)
        self.W -= np.diag(np.full(self.N, self.W[0][0]))

    def _iteration(self, signal):
        for i in range(self.N):
            H_i = 0
            for j in range(self.N):
                H_i += self.W[i][j] * signal[j]
            signal[i] = self.signum(H_i)
        return signal

    def associations(self, signal):  # распознавание образа
        X = signal.copy()  # текущее состояние
        for _ in range(self.K):
            x_before = X.copy()
            X = self._iteration(x_before.copy())

            if (x_before == X).all():  # выход из цикла, если значения стабилизировались
                return X
        return X

    def signum(self, a):  # функция активации
        return 1 if a >= 0 else -1

    @staticmethod
    def print_letter(brightness):
        Image.fromarray((brightness.reshape(image_size[0], image_size[1]) + 1) * 128).show()
