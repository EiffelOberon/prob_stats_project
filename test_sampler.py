import numpy as np
import os
import random

from sampler import *

if __name__ == '__main__':
    s = MetropolisSampler(100, 100, 1)
    print("seed: ", s.seed)
    for i in range(0, 5):
        f = s.next()
        print(s.X)
