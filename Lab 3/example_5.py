import numpy as np
import matplotlib.pyplot as plt
from neural_network import Kohonen, rand

"""
Показывает разные результаты для сигма и тетта с 3 входными весами
"""

train_data = np.append(rand.uniform(0, 0.2, (5000, 2)), rand.uniform(0.8, 1, (5000, 2)), axis=0)

plt.figure(figsize=(12, 12))

for i, learn_rate_0 in enumerate([0.001, 0.5, 0.99]):
    for j, radius_0 in enumerate([0.01, 1, 10]):
        kh = Kohonen(2, 10)
        kh.generate_W()
        kh.calculate(train_data, 5, learn_rate_0=learn_rate_0, radius_0=radius_0)
        plt.subplot(3, 3, i * 3 + j + 1)
        plt.scatter(*zip(*kh.W.reshape(1, 10 ** 2, 2)[0]))
        plt.title('$\eta$ = ' + str(learn_rate_0) + 
                                ', $\sigma^2$ = ' + str(radius_0))
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()