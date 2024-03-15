import os

import cupy as cp
from PIL import Image


def get_brightness(image_path):
    img = Image.open(image_path)
    img = img.convert("L")
    return cp.array(img.getdata()) / 255


dataset = []
letters = os.listdir('dataset')
for i in letters:
    fronts = os.listdir(f'dataset/{i}/')
    for j in fronts:
        brightness = get_brightness(f'dataset/{i}/{j}')
        dataset.append((brightness, ord(i.lower()) - ord('a')))


def get_data():
    return dataset
