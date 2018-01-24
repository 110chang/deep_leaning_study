import os
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append(os.pardir)
from functions import numerical_diff


def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


xx = np.arange(0.0, 20.0, 0.1)
yy = function_1(xx)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(xx, yy)
plt.show()

print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))
