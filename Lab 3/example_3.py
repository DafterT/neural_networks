import numpy as np
import matplotlib.pyplot as plt
from neural_network import Kohonen, rand

kh = Kohonen(3, 10)
train_data = rand.uniform(0, 1, (3000, 3))

fig, ax = plt.subplots(
    nrows=1, ncols=5, figsize=(12, 4), 
    subplot_kw=dict(xticks=[], yticks=[]))

ax[0].imshow(train_data.reshape(50, 60, 3))
ax[0].title.set_text('Исходные данные')

kh.generate_W()
ax[1].imshow(kh.W)
ax[1].title.set_text('Случайные веса')

for i, epochs in enumerate([1, 4, 5], start=2):
    kh.calculate(train_data, epochs)
    ax[i].imshow(kh.W)
    ax[i].title.set_text(f'Веса после обучения {[1, 5, 10][i - 2]}')

plt.tight_layout()
plt.show()