import os

import numpy as np
# import cupy as cp
from PIL import Image
from constants import image_size


def get_brightness(image_path):
    img = Image.open(image_path)
    img = img.convert("L")
    return ((np.round(np.array(img.getdata()) / 255) - 0.5) * 2).reshape(image_size[0] * image_size[1], 1)



dataset = []
letters = os.listdir(f'dataset/')
for i in letters:
    brightness = get_brightness(f'dataset/{i}')
    dataset.append(brightness)


def get_data():
    return dataset

def print_letter(brightness):
    Image.fromarray((brightness.reshape(image_size[0], image_size[1]) + 1) * 128).show()
