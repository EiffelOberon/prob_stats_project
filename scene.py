import numpy as np

from camera import *
from color import *
from ray import Ray
from settings import *
from sphere import *
from trace import *

class Scene:
    # keeps a set of variables about our scene to be rendered
    def __init__(self, sampling, width, height, sky_image, max_bounce):
        self.sampling = sampling
        self.max_bounce = max_bounce
        # dimension of our frame
        self.width = width
        self.height = height
        # camera
        self.camera = Camera(width, height)
        # object list
        object_list = [\
            Sphere(np.array([0.0, 0.0, -1.0]), 0.5, Metal(np.array([1.0, 1.0, 1.0]), 0.0)), \
            Sphere(np.array([1.5, 0.0, -2.0]), 0.5, Diffuse(np.array([0.0, 0.2, 0.6]))), \
            Sphere(np.array([-1.5, 0.0, -2.0]), 0.5, Metal(np.array([0.8, 0.2, 0.2]), 0.2)), \
            Sphere(np.array([0.0, -100.5, -1.0]), 100.0, Diffuse(np.array([1.0, 1.0, 1.0])))\
        ]
        # the world of objects
        self.world = Hittable_List(object_list)
        # HDRI environment texture we load at the start, this supplies radiance for lighting objects in the scene
        self.environment = sky_image
    
    def evaluate_environment(self, direction):
        # normalize the direction of the ray
        unit_direction = normalize(direction)
        # vector (y-up) to spherical coordinate
        theta = np.arctan2(unit_direction[2], unit_direction[0])
        phi = np.arcsin(unit_direction[1])
        # transform to [0,1] and then scale to resolution
        x = int(( theta / (2.0 * np.pi) + 0.5 ) * self.environment.shape[1])
        y = int((1.0 - ( phi / np.pi + 0.5 )) * self.environment.shape[0])
        # access pixel
        skyRadiance = self.environment[y][x]
        return np.array([skyRadiance[0], skyRadiance[1], skyRadiance[2]])