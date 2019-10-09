import numpy as np
import matplotlib.pyplot as plt

def trapezregel(func, a, b):
    return (b - a) / 2 * (func(a) + func(b))

def trapezregel_mInterval(func, a, b, m):
    x = np.linspace(a, b, num=m+1)
    limits = zip(x[:-1], x[1:])
    return sum([trapezregel(func, a, b) for (a, b) in limits])


testfunction = lambda x : x**3 + 4
print(trapezregel(testfunction, 2, 10))

m = range(1, 101)
y = [trapezregel_mInterval(testfunction, 2, 10, m) for m in m]

plt.plot(m, y)
plt.title('Trapezoidal convergence for $\\int_1^{10} x^3 + 4 \\, dx = 2528$')
plt.xlabel('Number of intervals')
plt.ylabel('Calculated integral')
plt.savefig('trapezregel.pdf')
plt.show()