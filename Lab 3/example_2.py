import numpy as np
from neural_network import Kohonen, rand, print_result

"""
Хз как описать, но выглядит круто
"""
n = 10

kh = Kohonen(2, n)
kh.generate_W()
train_data = np.append(rand.uniform(0, 0.2, (5000, 2)), rand.uniform(0.8, 1, (5000, 2)), axis=0)

print_result(n, kh, train_data, iterations=50)