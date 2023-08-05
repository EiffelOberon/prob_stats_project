import numpy as np
import random

from random import Random

class Sampler:
    def __init__(self, width, height, seed):
        self.width = width
        self.height = height
        self.seed = seed
        self.rng = Random()
        self.rng.seed(self.seed)

    def uniform(self):
        return self.rng.random()

    def next(self):
        return self.uniform()
        
class RandomNumberSampler(Sampler):
    def __init__(self, width, height, seed):
        super().__init__(width, height, seed)

class PrimarySample:
    def __init__(self):
        self.value = 0.0
        self.backup_value = 0.0
        self.last_modified_iter = 0
        self.last_modified_backup = 0
        
    def __repr__(self):
        return f"Value: {self.value}, Backup: {self.backup_value}"    
    
    def backup(self):
        self.backup_value = self.value
        # update backup iteration to the last modified iteration
        self.last_modified_backup = self.last_modified_iter

    def restore(self):
        self.value = self.backup_value
        # since we are using the backup value, our modified iteration goes back to 
        # the iteration when we stored the backup value
        self.last_modified_iter = self.last_modified_backup


class RadianceRecord:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radiance = np.array([0.0, 0.0, 0.0])

class MetropolisSampler(Sampler):
    def __init__(self, width, height, seed):
        super().__init__(width, height, seed)
        self.X = np.array([])
        self.current_iter = 0
        self.large_step = True
        self.last_large_step_iter = 0
        self.sample_index = 0
        self.accepted = 0
        self.rejected = 0
        self.current_record = RadianceRecord(0, 0)

    def __str__(self):
        return f"{self.accepted}({self.rejected})"    
    
    def start_iteration(self):
        self.sample_index = 0
        self.current_iter += 1
        # probability of 0.25 to sample large step
        self.large_step = self.uniform() < 0.25

    def mutate(self, idx):
        s1 = 1.0 / self.width
        s2 = 0.1
        if(idx >= 2):
            s1 = 1.0 / 1024.0
            s2 = 1.0 / 64.0
        elif(idx == 1):
            s1 = 1.0 / self.height
            s2 = 0.1
        if(self.X[idx].last_modified_iter < self.last_large_step_iter):
            self.X[idx].value = self.uniform()
            # update last modified iteration to the last large step iteration
            self.X[idx].last_modified_iter = self.last_large_step_iter
        if(self.large_step == True):
            self.X[idx].backup()
            self.X[idx].value = self.uniform()
        else:
            n_diff = self.current_iter - self.X[idx].last_modified_iter - 1
            # mutate n-1 times first, then mutate the nth time and keep n-1th mutation
            # as the back up
            if(n_diff > 0):
                val = self.X[idx].value
                while(n_diff > 0):
                    n_diff -= 1
                    val = self.mutateRN(val, s1, s2)
                self.X[idx].value = val
                self.X[idx].last_modified_iter = self.current_iter - 1
            self.X[idx].backup()
            self.X[idx].value = self.mutateRN(self.X[idx].value, s1, s2)
        self.X[idx].last_modified_iter = self.current_iter
        return
    
    def next(self):
        if(self.sample_index >= self.X.size):
            self.X = np.append(self.X, np.array([PrimarySample()]))
        self.mutate(self.sample_index)
        self.sample_index += 1
        return self.X[self.sample_index - 1].value

    def mutateRN(self, x, s1, s2):
        r = self.uniform()
        if(r < 0.5):
            r = r * 2.0
            x = x + s2 * np.exp(-np.log(s2 / s1) * r)
            if(x > 1.0):
                x = x - 1.0
        else:
            r = (r - 0.5) * 2.0
            x = x - s2 * np.exp(-np.log(s2 / s1) * r)
            if(x < 0.0):
                x = x + 1.0
        return x
    
    def accept(self):
        if(self.large_step):
            self.last_large_step_iter = self.current_iter
        self.accepted += 1
        return
    
    def reject(self):
        for i in range(0,self.X.size):
            if(self.X[i].last_modified_iter == self.current_iter):
                self.X[i].restore()
        self.rejected += 1
        self.current_iter -= 1
        return