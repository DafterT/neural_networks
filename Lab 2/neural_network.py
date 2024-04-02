import string
from time import sleep

import numpy as np
from PIL import Image

from constants import image_size
from read_dataset import dataset


class Hopfild:

    def __init__(self, k):  # инициализация
        self.N = image_size[0] * image_size[1]  # количество нейронов
        self.K = k  # максимальное количество эпох распознавания сигнала
        self.W = np.zeros((self.N, self.N))  # матрица взаимодействий (весов)

    def remember(self, M):  # метод запоминания образов
        for X in M:
            self.W += X.dot(X.T)
        self.W -= np.diag(np.full(self.N, self.W[0][0]))

    def associations(self, signal):  # распознавание образа
        X = signal.copy()  # текущее состояние
        for i in range(self.K):
            x_before = X.copy()  # предыдущее состояние
            a_i = np.dot(self.W, X)
            X = self.signum(a_i)

            if (x_before == X).all():  # выход из цикла, если значения стабилизировались
                return X
        print('!')
        return X

    def signum(self, a):  # функция активации
        return np.array([1 if i >= 0 else -1 for i in a])

    @staticmethod
    def print_letter(brightness):
        Image.fromarray((brightness.reshape(image_size[0], image_size[1]) + 1) * 128).show()


l = Hopfild(100)
l.remember(dataset[::8])
print(string.ascii_lowercase[::8])
l.print_letter(dataset[0])
l.print_letter(l.associations(dataset[0]))
# l.print_letter(dataset[10])
# sleep(3)
# l.print_letter(l.associations(dataset[-1]))
