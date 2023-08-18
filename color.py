import numpy as np
from sys import platform

def linear_to_srgb(c):
    r = np.clip(np.power(c[0], 1.0/2.2), 0.0, 1.0)
    g = np.clip(np.power(c[1], 1.0/2.2), 0.0, 1.0)
    b = np.clip(np.power(c[2], 1.0/2.2), 0.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def linear_to_srgb_no_gamma(c):
    r = np.clip(c[0], 0.0, 1.0)
    g = np.clip(c[1], 0.0, 1.0)
    b = np.clip(c[2], 0.0, 1.0)
    return (int(r * 255), int(g * 255), int(b * 255))

def srgb_to_linear(c):
    r = pow(c[0] / 255.0, 2.2)
    g = pow(c[1] / 255.0, 2.2)
    b = pow(c[2] / 255.0, 2.2)
    return np.array([r,g,b])

def luminance(c):
    weights = np.array([0.2125, 0.7154, 0.0721])
    return np.dot(c, weights)