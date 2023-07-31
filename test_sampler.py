import numpy as np
import os
import random

from sampler import *

if __name__ == '__main__':
    s = RandomNumberSampler(100, 100, 1)
    for i in range(0, 100000):
        print(s.seed)
        f = s.next2()
        if(f[0] < 0.0 or f[0] > 1.0 or f[1] < 0.0 or f[1] > 1.0):
            print("Unexpected: ", f)
            break
        print(f)