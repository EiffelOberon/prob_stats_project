import numpy as np
import random

from hittable import Hittable, HitRecord
from hittable_list import Hittable_List
from material import *
from path import Path
from scene import Scene, Sampling
from sphere import *
from util import *

def trace(scene, sampler, ray, path):
    hit_record = HitRecord()
    if scene.world.hit(ray, 0.0001, 10000000.0, hit_record) > 0:
        # increment bounce
        path.bounce = path.bounce + 1
        # limit bounce
        if path.bounce < scene.max_bounce:
            # initialize ray for our scattering from current bounce
            scattered_ray = Ray(origin=np.array([0.0, 0.0, 0.0]), direction=np.array([0.0, 0.0, 0.0]))
            # sample according to material's BSDF
            reflectance = hit_record.material.sample(scene, sampler, path, ray, hit_record, scattered_ray)
            # trace next ray
            return trace(scene, sampler, scattered_ray, path) * reflectance
        else:
            # if exceed bounce count then there is no radiance contribution
            return np.array([0.0, 0.0, 0.0])
    else:
        return scene.evaluate_environment(ray.direction)