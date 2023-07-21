import numpy as np

from util import *

class Hittable():
    def __init__(self, origin, direction):
        return

class HitRecord:
    def __init__(self, t=0.0, p=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, 0.0, 0.0])):
        self.t = t
        self.p = p
        self.normal = normal
        self.material = None