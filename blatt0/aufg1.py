import numpy as np
import matplotlib.pyplot as plt

def fibonacci(n):
    return 1 if n <= 1 else fibonacci(n-1) + fibonacci(n-2)

def plot_fibonacci(max_n):
    x = range(max_n + 1)
    y = [fibonacci(n) for n in x]
    plt.plot(x, y)
    plt.title('First '+ str(max_n) + ' Fibonacci Numbers')
    plt.savefig('fibonacci_plot.pdf')
    plt.show()

plot_fibonacci(30)

x = np.arange(0, 21, 0.1)
y = [80 + np.cos(5*x/np.pi) for x in x]
plt.plot(x, y, label='MyWeight')

y = [x + 70 for x in x]
plt.plot(x, y, label='StraightOne')

y = [50 + np.floor(x) for x in x]
plt.plot(x, y, label='StairwayToHeaven')

y = [70 - x if x < 10 else 50 + x for x in x]
plt.plot(x, y, label='V-Shape')

plt.legend()
plt.savefig('4functions.pdf')
plt.show()

plt.figure(figsize=(8, 7))

plt.subplot(221)
x = np.arange(0, 21, 0.1)
y = [80 + np.cos(5*x/np.pi) for x in x]
plt.plot(x, y)
plt.title('MyWeight')

plt.subplot(222)
y = [x + 70 for x in x]
plt.plot(x, y)
plt.title('StraightOne')

plt.subplot(223)
y = [50 + np.floor(x) for x in x]
plt.plot(x, y)
plt.title('StairwayToHeaven')

plt.subplot(224)
y = [70 - x if x < 10 else 50 + x for x in x]
plt.plot(x, y)
plt.title('V-Shape')

plt.savefig('4functions-subplots.pdf')
plt.show()

