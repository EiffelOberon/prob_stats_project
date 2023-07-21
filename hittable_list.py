import numpy as np

from hittable import Hittable, HitRecord
from util import *
from sphere import *

class Hittable_List(Hittable):
    def __init__(self, hittable_objects):
        self.hittable_objects = hittable_objects

    def hit(self, ray, t_min, t_max, hit_record):
        # keep a temporary hit record
        temp_hit_rec = HitRecord()
        any_hit = False
        closest_distance = t_max
        for item in self.hittable_objects:
            if item.hit(ray, t_min, closest_distance, temp_hit_rec):
                any_hit = True
                closest_distance = temp_hit_rec.t
                hit_record.t = temp_hit_rec.t
                hit_record.p = temp_hit_rec.p
                hit_record.normal = temp_hit_rec.normal
                hit_record.material = temp_hit_rec.material
        return any_hit

    def has_any_hit(self, ray, t_min, t_max):
        # keep a temporary hit record
        any_hit = False
        closest_distance = t_max
        temp_hit_rec = HitRecord()
        for item in self.hittable_objects:
            if item.hit(ray, t_min, closest_distance, temp_hit_rec):
                any_hit = True
        return any_hit