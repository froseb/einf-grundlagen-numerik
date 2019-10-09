import numpy as np
import matplotlib.pyplot as plt

def fibonacci(n):
    return 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

def plot_fibonacci(max_n):
    x = range(max_n)
    y = [fibonacci(n) for n in x]
    plt.plot(x, y)

plot_fibonacci(30)
plt.show()