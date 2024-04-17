import numpy as np
import matplotlib.pyplot as plt
from neural_network import Kohonen, rand

"""
Показывает разные результаты для сигма и тетта с 3 входными весами
"""

train_data = rand.uniform(0, 1, (3000, 3))

fig, ax = plt.subplots(
    nrows=3, ncols=3, figsize=(12, 12), 
    subplot_kw=dict(xticks=[], yticks=[]))

for i, learn_rate_0 in enumerate([0.001, 0.5, 0.99]):
    for j, radius_0 in enumerate([0.01, 1, 10]):
        kh = Kohonen(3, 10)
        kh.generate_W()
        kh.calculate(train_data, 5, learn_rate_0=learn_rate_0, radius_0=radius_0)
        ax[i][j].imshow(kh.W)
        ax[i][j].title.set_text('$\eta$ = ' + str(learn_rate_0) + 
                                ', $\sigma^2$ = ' + str(radius_0))
plt.show()