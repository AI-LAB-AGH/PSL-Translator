import numpy as np

print(np.dot(np.array([[1,2],[2,3]]),np.array([[0,1],[1,1]])))

def f(values):
    values[0] = 42

values = [1,2,3]
f(values)
print(values)

print(bool([]))