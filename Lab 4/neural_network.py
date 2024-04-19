import numpy as np

rand = np.random.RandomState(0)


class ART_1:
    def __init__(self, input_layer, output_layer, p):
        self.input_layer = input_layer
        self.output_layer = output_layer
        self.p = p
        self.W = np.ones((input_layer, output_layer))

    def calculate(self, in_x):
        enabled_nodes = np.ones(self.output_layer)
        while True:
            y = (self.W / (0.5 + np.sum(self.W))).T.dot(in_x) * enabled_nodes
            i = np.argmax(y)
            r = np.sum(self.W[:, i] * in_x) / (np.sum(np.abs(in_x)))
            if r > self.p:
                self.W[:, i] = self.W[:, i] * in_x
                return i
            elif np.sum(enabled_nodes) > 0:
                enabled_nodes[i] = 0
            else:
                self.output_layer += 1
                self.W = np.c_[self.W, np.ones(self.input_layer)]
                enabled_nodes = np.zeros(self.output_layer)
                enabled_nodes[-1] = 1


def main():
    pass


if __name__ == '__main__':
    main()
