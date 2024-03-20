import random

import keyboard
from matplotlib import pyplot as plt

from activation_functions import *
from read_dataset import get_data

INPUT_DIM = 28 * 28
OUT_DIM = 26
H_DIM = 28 * 28 * 26 // 10
ALPHA = 0.0001
NUM_EPOCHS = 20


def softmax(t):
    out = calc.exp(t)
    return out / calc.sum(out)


def softmax_deriv(t):
    return softmax(t) * (1 - softmax(t))


def to_full(y, num_classes):
    y_full = calc.zeros((num_classes, 1))
    y_full[y, 0] = 1
    return y_full


def predict(x, W1, b1, W2, b2, relu):
    t1 = W1 @ x + b1
    h1 = relu(t1)
    t2 = W2 @ h1 + b2
    z = softmax(t2)
    return z


def calc_accuracy(check_correct, W1, b1, W2, b2, relu):
    correct = 0
    for x, y in check_correct:
        x = x[:, None]
        z = predict(x, W1, b1, W2, b2, relu)
        y_pred = calc.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(check_correct)
    return acc * 100


def train(dataset, check_correct, W1, b1, W2, b2, relu, relu_deriv):
    loss_arr = []
    #for ep in range(NUM_EPOCHS):
    accuracy = 0
    ep = 0
    while accuracy < 95 and (not keyboard.is_pressed('ctrl') or not keyboard.is_pressed('alt')):
        random.shuffle(dataset)
        ep += 1
        for i in range(len(dataset)):
            x, y = dataset[i]
            x = x[:, None]
            # Forward
            t1 = W1 @ x + b1
            h1 = relu(t1)
            t2 = W2 @ h1 + b2
            h2 = softmax(t2)
            # Backward
            y_full = to_full(y, OUT_DIM)
            d3 = (h2 - y_full)
            d2 = W2.T @ d3
            dW2 = ((d3 * softmax_deriv(t2)) @ h1.T) * ALPHA
            dW1 = ((d2 * relu_deriv(t1)) @ x.T) * ALPHA
            db2 = (d3 * softmax_deriv(t2)) * ALPHA
            db1 = (d2 * relu_deriv(t1)) * ALPHA
            # Update
            W2 = W2 - dW2
            W1 = W1 - dW1
            b2 = b2 - db2
            b1 = b1 - db1
            if i % 100 == 0:
                print(ep, i, calc.argmax(h2), y)
        accuracy = calc_accuracy(check_correct, W1, b1, W2, b2, relu)
        loss_arr.append(float(accuracy))
        print("Accuracy:", accuracy)
    calc.savetxt(f'data/{relu.__name__}_W1.txt', W1)
    calc.savetxt(f'data/{relu.__name__}_b1.txt', b1)
    calc.savetxt(f'data/{relu.__name__}_W2.txt', W2)
    calc.savetxt(f'data/{relu.__name__}_b2.txt', b2)
    return loss_arr


def main():
    dataset = get_data()
    random.shuffle(dataset)
    check_correct = dataset[:int(calc.round(len(dataset) * 0.1))]
    dataset = dataset[int(calc.ceil(len(dataset) * 0.1)):]

    W1 = calc.random.rand(H_DIM, INPUT_DIM)
    b1 = calc.random.rand(H_DIM, 1)
    W2 = calc.random.rand(OUT_DIM, H_DIM)
    b2 = calc.random.rand(OUT_DIM, 1)

    W1 = (W1 - 0.5) * 2 * calc.sqrt(1 / H_DIM)
    b1 = (b1 - 0.5) * 2 * calc.sqrt(1 / H_DIM)
    W2 = (W2 - 0.5) * 2 * calc.sqrt(1 / OUT_DIM)
    b2 = (b2 - 0.5) * 2 * calc.sqrt(1 / OUT_DIM)
    for (act, act_deriv), name in [(get_linear_max(), 'max(x, 0)')]:
        accuracy = calc_accuracy(check_correct, W1, b1, W2, b2, act)
        print("Accuracy:", accuracy)
        x = [
            accuracy, *train(
                dataset.copy(), check_correct.copy(),
                W1.copy(), b1.copy(), W2.copy(), b2.copy(),
                act, act_deriv)
        ]
        with open('log.log', 'r+') as f:
            f.seek(0, 2)
            f.write(f'{name:} {repr(x)}\n')
        plt.plot(x, label=name)
    plt.legend()
    plt.savefig(f'./img/result.png')
    plt.show()


if __name__ == '__main__':
    main()
