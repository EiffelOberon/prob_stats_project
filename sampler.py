import numpy as np
import random

class Sampler:
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed
        self.current_iteration = 0
        self.sample_index = 0
        self.X = []

    def uniform(self):
        self.seed = (1103515245 * self.seed + 12345)
        self.seed = self.seed % 0xFFFFFFFF
        return float(self.seed) / float(0xFFFFFFFF)

    def next(self):
        return self.uniform()
        
    def next2(self):
        return np.array([self.uniform(), self.uniform()])

class RandomNumberSampler(Sampler):
    def __init__(self, width, height, seed):
        super().__init__(width, height, seed)
