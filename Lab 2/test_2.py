from constants import image_size
from neural_network import Hopfild
from read_dataset import get_data, print_letter

"""
Этот тест демонстрирует, что при сохранении близких значений получаются химеры
"""

l = Hopfild(image_size[0] * image_size[1], 100)
dataset = get_data()
l.remember(dataset)
print_letter(dataset[0])
print_letter(l.associations(dataset[0]))
