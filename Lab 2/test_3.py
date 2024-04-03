from generate_dataset_2 import get_data, image_size, swap_elements, parse_3_images
from neural_network import Hopfild

"""
Этот тест демонстрирует, корректную работу нейронной сети на меньшем размере букв
"""

l = Hopfild(image_size[0] * image_size[1], 100)
dataset = get_data()
l.remember(dataset)
x = swap_elements(dataset[0], int(image_size[0] * image_size[1] * 0.8))
parse_3_images(x, dataset[0], l.associations(x))
l.print_energy()
