import numpy as np
from neural_network import Kohonen, rand, print_result

"""
Пример показывает решетку n на n, если все входные данные случайны
"""
n = 10

kh = Kohonen(2, n)
kh.generate_W()
train_data = rand.uniform(0, 1, (10**4, 2))

print_result(n, kh, train_data)