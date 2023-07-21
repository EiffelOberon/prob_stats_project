import numpy as np

from hittable import Hittable
from ray import *
from util import *

class Sphere(Hittable):
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material
    
    def hit(self, ray, t_min, t_max, hit_record):
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius*self.radius
        discriminant = b*b - a*c
        if discriminant > 0.0:
            temp = (-b - np.sqrt(discriminant))/a
            if t_min < temp < t_max:
                hit_record.t = temp
                hit_record.p = ray.origin + ray.direction * hit_record.t
                hit_record.normal = (hit_record.p - self.center)/self.radius
                hit_record.normal = normalize(hit_record.normal)
                hit_record.material = self.material
                return True
            temp = (-b + np.sqrt(discriminant))/a
            if t_min < temp < t_max:
                hit_record.t = temp
                hit_record.p = ray.origin + ray.direction * hit_record.t
                hit_record.normal = (hit_record.p - self.center)/self.radius
                hit_record.normal = normalize(hit_record.normal)
                hit_record.material = self.material
                return True
        else:
            return False
        return