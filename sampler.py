import numpy as np
import random

from random import Random

class Sampler:
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed
        self.current_iteration = 0
        self.sample_index = 0
        self.X = []
        self.rng = Random()
        self.rng.seed(self.seed)

    def uniform(self):
        return self.rng.random()

    def next(self):
        return self.uniform()
        
    def next2(self):
        return np.array([self.uniform(), self.uniform()])

class RandomNumberSampler(Sampler):
    def __init__(self, width, height, seed):
        super().__init__(width, height, seed)
