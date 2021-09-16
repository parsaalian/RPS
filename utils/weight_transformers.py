import numpy as np


def coth_transformer(weight):
    COTH1 = 1 / np.tanh(1)
    return abs(1 / np.tanh(weight)) - COTH1