import numpy as np

from ray import Ray

class Camera:
    def __init__(self, width, height):
        aspect_ratio = float(width) / float(height)
        self.horizontal = np.array([4.0, 0.0, 0.0])
        self.vertical = np.array([0.0, self.horizontal[0] / aspect_ratio, 0.0])
        self.lower_left = np.array([0.0 - self.horizontal[0] * 0.5, 0.0 - self.vertical[1] * 0.5, -1.0])
        self.origin = np.array([0.0, 0.0, 2.0])

    def get_ray(self, u, v):
        return Ray(origin = self.origin, direction = self.lower_left + self.horizontal * u + self.vertical * v - self.origin)
    