import numpy as np
import math

def linear_kernel():
    def kernel(x, y):
        return np.dot(x, y) + 1
    return kernel

def polynomial_kernel(p):
    def kernel(x, y):
        return (np.dot(x, y) + 1)**p
    return kernel

def radial_kernel(sigma):
    def kernel(x, y):
        x = np.array(x)
        y = np.array(y)
        norm2 = np.linalg.norm(x-y, 2)**2
        return math.exp(-norm2 / (2* sigma**2) )
    return kernel

def sigmoid_kernels(k, delta):
    def kernel(x, y):
        return math.tanh(k* np.dot(x,y) - delta)
    return kernel