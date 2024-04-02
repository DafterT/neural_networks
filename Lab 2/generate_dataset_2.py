import numpy as np

from_symbol_to_number = {'#': 1, '_': -1}
from_number_to_symbol = {1: '#', -1: '_'}

t = "##########" \
    "##########" \
    "____##____" \
    "____##____" \
    "____##____" \
    "____##____" \
    "____##____" \
    "____##____" \
    "____##____" \
    "____##____"

n = "##______##" \
    "##______##" \
    "##______##" \
    "##______##" \
    "##########" \
    "##########" \
    "##______##" \
    "##______##" \
    "##______##" \
    "##______##"

k = "##______##" \
    "##_____##_" \
    "##____##__" \
    "##__##____" \
    "####______" \
    "####______" \
    "##__##____" \
    "##____##__" \
    "##_____##_" \
    "##______##"
image_size = [10, 10]


def create_image(base, dic=from_symbol_to_number):  # Создает бинарный вектор образа
    a = np.array([dic[i] for i in base]).reshape(image_size[0] * image_size[1], 1)
    return a


def parse_image(img, n=10, dic=from_number_to_symbol):  # Выводит образ по бинарному вектору
    for i in range(0, len(img), n):
        print(''.join([dic[i[0]] for i in img[i:(i + n)]]))


def parse_3_images(img1, img2, img3, n=10, dic=from_number_to_symbol):
    for i in range(0, len(img1), n):
        print(''.join([dic[i[0]] for i in img1[i:(i + n)]]), end='  ')
        print(''.join([dic[i[0]] for i in img2[i:(i + n)]]), end='  ')
        print(''.join([dic[i[0]] for i in img3[i:(i + n)]]))


def get_data():
    return np.array([create_image(n),
                     create_image(t),
                     create_image(k)])


def swap_elements(arr, n):
    # Генерируем случайные индексы для выбора элементов
    indices = np.random.choice(len(arr), n, replace=False)

    # Создаем копию массива для изменений
    arr_copy = arr.copy()

    # Меняем местами выбранные элементы
    arr_copy[indices] = np.roll(arr_copy[indices], 1)

    return arr_copy
