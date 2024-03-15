import random

from matplotlib import pyplot as plt

from activation_functions import *
from read_dataset import get_data

INPUT_DIM = 28 * 28
OUT_DIM = 26
H_DIM = 28 * 28 * 26
ALPHA = 0.0001
NUM_EPOCHS = 2


def softmax(t):
    out = calc.exp(t)
    return out / calc.sum(out)


def sparse_cross_entropy(z, y):
    return -calc.log(z[0, y])


def to_full(y, num_classes):
    y_full = calc.zeros((1, num_classes))
    y_full[0, y] = 1
    return y_full


def predict(x, W1, b1, W2, b2, relu):
    t1 = x @ W1 + b1
    h1 = relu(t1)
    t2 = h1 @ W2 + b2
    z = softmax(t2)
    return z


def calc_accuracy(check_correct, W1, b1, W2, b2, relu):
    correct = 0
    for x, y in check_correct:
        z = predict(x, W1, b1, W2, b2, relu)
        y_pred = calc.argmax(z)
        if y_pred == y:
            correct += 1
    acc = correct / len(check_correct)
    return acc * 100


def train(dataset, check_correct, W1, b1, W2, b2, relu, relu_deriv):
    loss_arr = []
    for ep in range(NUM_EPOCHS):
        random.shuffle(dataset)
        for i in range(len(dataset)):
            x, y = dataset[i]
            # Forward
            t1 = x @ W1 + b1
            h1 = relu(t1)
            t2 = h1 @ W2 + b2
            z = softmax(t2)
            E = calc.sum(sparse_cross_entropy(z, y))
            # Backward
            y_full = to_full(y, OUT_DIM)
            dE_dt2 = z - y_full
            dE_dW2 = h1.T @ dE_dt2
            dE_db2 = dE_dt2
            dE_dh1 = dE_dt2 @ W2.T
            dE_dt1 = dE_dh1 * relu_deriv(t1)
            dE_dW1 = x[:, None] @ dE_dt1
            dE_db1 = dE_dt1
            # Update
            W1 = W1 - ALPHA * dE_dW1
            b1 = b1 - ALPHA * dE_db1
            W2 = W2 - ALPHA * dE_dW2
            b2 = b2 - ALPHA * dE_db2
            if i % 100 == 0:
                print(ep, i, float(E))
        accuracy = calc_accuracy(check_correct, W1, b1, W2, b2, relu)
        loss_arr.append(float(accuracy))
        print("Accuracy:", accuracy)
    return loss_arr


def main():
    dataset = get_data()
    random.shuffle(dataset)
    check_correct = dataset[:int(calc.round(len(dataset) * 0.1))]
    dataset = dataset[int(calc.ceil(len(dataset) * 0.1)):]

    W1 = calc.random.rand(INPUT_DIM, H_DIM)
    b1 = calc.random.rand(1, H_DIM)
    W2 = calc.random.rand(H_DIM, OUT_DIM)
    b2 = calc.random.rand(1, OUT_DIM)

    W1 = (W1 - 0.5) * 2 * calc.sqrt(1 / INPUT_DIM)
    b1 = (b1 - 0.5) * 2 * calc.sqrt(1 / INPUT_DIM)
    W2 = (W2 - 0.5) * 2 * calc.sqrt(1 / H_DIM)
    b2 = (b2 - 0.5) * 2 * calc.sqrt(1 / H_DIM)
    calc.random.seed(0)
    for (act, act_deriv), name in get_functions()[:1]:
        accuracy = calc_accuracy(check_correct, W1, b1, W2, b2, act)
        print("Accuracy:", accuracy)
        x = [accuracy, *train(dataset.copy(), check_correct.copy(), W1, b1, W2, b2, act, act_deriv)]
        print(x)
        with open('log.log', 'r+') as f:
            f.seek(0, 2)
            f.write(f'{name:} {str(x)}\n')
        plt.plot(x, label=name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
