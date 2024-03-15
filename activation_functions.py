import cupy as calc


def _sigmoid(t):
    return 1 / (1 + calc.exp(-t))


def _sigmoid_deriv(t):
    return calc.exp(t) / (calc.exp(2 * t) + 2 * calc.exp(t) + 1)


def get_sigmoid():
    return _sigmoid, _sigmoid_deriv


def _linear_max(t):
    return calc.maximum(t, 0)


def _linear_max_deriv(t):
    return (t >= 0).astype(float)


def get_linear_max():
    return _linear_max, _linear_max_deriv


def _linear(t):
    return t


def _linear_deriv(t):
    return 1


def get_linear():
    return _linear, _linear_deriv


def _tanh(t):
    return (calc.exp(t) - calc.exp(-t)) / (calc.exp(t) + calc.exp(-t))


def _tanh_deriv(t):
    return 1 - (calc.exp(4 * t) - 2 * calc.exp(2 * t) + 1) / (calc.exp(4 * t) + 2 * calc.exp(2 * t) + 1)


def get_tanh():
    return _tanh, _tanh_deriv


def get_functions():
    functions = []
    functions.append((get_tanh(), 'y = Tanh(x)'))
    functions.append((get_linear(), 'y = x'))
    functions.append((get_linear_max(), 'y = max(x, 0)'))
    functions.append((get_sigmoid(), 'y = sigm(x)'))
    return functions
