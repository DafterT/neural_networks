import numpy as np
import matplotlib.pyplot as plt
from neural_network import Kohonen, rand

kh = Kohonen(3, 10)
train_data = rand.uniform(0, 1, (3000, 3))

fig, ax = plt.subplots(
    nrows=1, ncols=3, figsize=(12, 3.5), 
    subplot_kw=dict(xticks=[], yticks=[]))
ax[0].imshow(train_data.reshape(50, 60, 3))
ax[0].title.set_text('Исходные данные')

kh.generate_W()
ax[1].imshow(kh.W)
ax[1].title.set_text('Случайно сгенерированные веса')

kh.calculate(train_data, 10)
ax[2].imshow(kh.W)
ax[2].title.set_text('Веса после обучения')
plt.show()