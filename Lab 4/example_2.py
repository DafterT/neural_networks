import numpy as np
from matplotlib import pyplot as plt

from neural_network import ART_1

art_1 = ART_1(25, 1, 0.3)
input_1 = [
    1, 0, 0, 0, 0,
    0, 1, 0, 0, 0,
    0, 0, 1, 0, 0,
    0, 0, 0, 1, 0,
    0, 0, 0, 0, 1
]
input_2 = [
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    0, 0, 1, 0, 0,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1
]
input_3 = [
    1, 0, 0, 0, 1,
    0, 1, 0, 1, 0,
    1, 1, 1, 1, 1,
    0, 1, 0, 1, 0,
    1, 0, 0, 0, 1
]
input_4 = [
    1, 0, 0, 0, 1,
    1, 1, 0, 1, 1,
    1, 1, 1, 1, 1,
    1, 1, 0, 1, 1,
    1, 0, 0, 0, 1
]
art_1.calculate(input_1)
W_1_temp = art_1.W.copy()
art_1.calculate(input_2)
W_2_temp = art_1.W.copy()
art_1.calculate(input_3)
W_3_temp = art_1.W.copy()
art_1.calculate(input_4)
W_4_temp = art_1.W.copy()

figure, axes = plt.subplots(4, 5, figsize=(12, 6))

for ax, array in zip(axes, [input_1, input_2, input_3, input_4]):
    ax[0].imshow(np.array(array).reshape(5, 5), cmap='binary', interpolation='nearest')
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
for i, W in enumerate([W_1_temp, W_2_temp, W_3_temp, W_4_temp], start=1):
    for ax, array in zip(axes, W.T):
        if array is None:
            array = np.zeros(5, 5)
        ax[i].imshow(np.array(array).reshape(5, 5), cmap='binary', interpolation='nearest')
        ax[i].get_xaxis().set_visible(False)
        ax[i].get_yaxis().set_visible(False)
[axes[-1][i].axis('off') for i in range(1, 5)]
[axes[-2][i].axis('off') for i in range(1, 5)]
[axes[-3][i].axis('off') for i in range(1, 4)]
plt.show()
