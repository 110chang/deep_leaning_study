import numpy as np
from functions import numerical_gradient


def function_2(x):
    return x[0] ** 2 + x[1] ** 2


print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))
