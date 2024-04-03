import numpy as np

from neural_network import Hopfild

"""
Этот тест демонстрирует, что при запоминании 1 объекта, мы получаем его инверсную копию
"""

l = Hopfild(4, 100)
l.remember([np.array([1, 1, -1, -1]).reshape(4, 1)])
print(l.associations(np.array([1, -1, -1, 1])))
print(l.associations(np.array([1, 1, -1, -1])))
l.print_energy()
