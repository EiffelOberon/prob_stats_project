import numpy as np
import random

from util import *
from color import *
from hittable_list import Hittable_List
from path import Path
from settings import *

def reflect(v, normal):
    return v - normal * (np.dot(v, normal) * 2)

class TangentBasis:
    # basis vectors for transforming between world space vector to
    # tangent space defined by a surface's normal, tangent and bitangent
    def __init__(self, normal):
        self.normal = normalize(normal)
        if(self.normal[0] != self.normal[1] or self.normal[0] != self.normal[2]):
            self.tangent = np.cross(self.normal, np.array([1.0, 1.0, 1.0]))
        else:
            self.tangent = np.cross(self.normal, np.array([-1.0, 1.0, 1.0]))
        self.tangent = normalize(self.tangent)
        self.bitangent = normalize(np.cross(self.tangent, self.normal))
    # transform to the tangent space
    def transform_to(self, v):
        return np.array([np.dot(self.tangent, v), np.dot(self.bitangent, v), np.dot(self.normal, v)])
    # transform to world space
    def transform_from(self, v):
        return self.tangent * v[0] + self.bitangent * v[1] + self.normal * v[2]

class Material:

    # abstract method inherited by subclasses
    def importance_sample(self, scene, path, ray, hit_record, scattered_ray):
        pass

    def importance_resample(self, scene, path, ray, hit_record, scattered_ray):
        count = 8
        total_weight = 0.0
        samples = []
        # result
        index = 0
        target_pdf = 0.0
        for i in range(0, count):
            result = self.importance_sample(scene, path, ray, hit_record, scattered_ray)
            samples.append(result[0])
            # multiply reflectance by environment radiance
            g = luminance(np.multiply(scene.evaluate_environment(scattered_ray.direction), samples[i]))
            g_over_f = g
            if(result[1] > 0.0):
                g_over_f = g / result[1]
            total_weight = total_weight + g_over_f
            # offset
            offset = path.y * scene.width + path.x
            offset = offset * int(RandomNumber.RANDOM_COUNT) * scene.max_bounce
            offset = offset + int(RandomNumber.RANDOM_COUNT) * path.bounce
            # random numbers
            r = scene.rng[offset + int(RandomNumber.RANDOM_RIS_1) + i]
            if( r * total_weight < g_over_f):
                 index = i
                 target_pdf = g
        recip = ( target_pdf * count )
        if(recip > 0.0):
            recip = 1.0 / recip
        else:
            recip = 0.0
        return samples[index] * recip * total_weight

    # generic sample function calling sampling functions for subclasses
    def sample(self, scene, path, ray, hit_record, scattered_ray):
        if(scene.sampling == Sampling.IMPORTANCE_SAMPLING):
            # brdf, pdf
            result = self.importance_sample(scene, path, ray, hit_record, scattered_ray)
            reflectance = 0.0
            if(result[1] > 0.0):
                reflectance = result[0] / result[1]
            return reflectance
        else:
            return self.importance_resample(scene, path, ray, hit_record, scattered_ray)

class Diffuse(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def importance_sample(self, scene, path, ray, hit_record, scattered_ray):
        # random numbers
        r = scene.get_brdf_r(path.x, path.y, path.bounce)
        # cosine hemisphere sampling
        cos_theta = np.sqrt(r[0])
        sin_theta = np.sqrt(1.0 - r[0]) 
        cos_phi = np.cos(2.0 * np.pi * r[1])
        sin_phi = np.sin(2.0 * np.pi * r[1])
        # convert spherical coordinates to vector in our tangent space of the surface
        vec_tangent = np.array([cos_phi * sin_theta, sin_phi * sin_theta, cos_theta])
        # transform to world space
        basis = TangentBasis(hit_record.normal)
        vec = basis.transform_from(vec_tangent)
        # write position and scattered direction to result
        scattered_ray.origin = hit_record.p
        scattered_ray.direction = normalize(vec)
        # compute the importance sampling PDF
        inv_pi = 1.0 / np.pi 
        pdf = inv_pi
        # calculate reflectance
        return ((self.albedo * inv_pi  * np.dot(vec_tangent, np.array([0.0, 0.0, 1.0]))), pdf)
    
class Metal(Material):
    def __init__(self, albedo, roughness):
        self.albedo = albedo
        if roughness < 1.0:
            self.roughness = roughness
        else: 
            self.roughness = 1.0

    def importance_sample(self, scene, path, ray, hit_record, scattered_ray):
        # random numbers
        r = scene.get_brdf_r(path.x, path.y, path.bounce)
        # clamp roughness squared for numerical precision of the distribution function
        alpha2 = max(self.roughness * self.roughness, 0.00001)
        # sample microfacet normal (also known as half vector)
        cos_theta = np.sqrt(max(0.0, ( 1.0 - r[0] ) / ( 1.0 + ( alpha2 - 1.0 ) * r[0] )))
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        cos_phi = np.cos(2.0 * np.pi * r[1])
        sin_phi = np.sin(2.0 * np.pi * r[1])
        microfacet_normal_tangent = normalize(np.array([sin_theta * cos_phi, sin_theta * sin_phi, cos_theta]))
        # transform microfacet normal to world space
        basis = TangentBasis(hit_record.normal)
        microfacet_normal_world = basis.transform_from(microfacet_normal_tangent)
        outgoing_vector = -normalize(ray.direction)
        outgoing_vector_tangent = normalize(basis.transform_to(outgoing_vector))
        # write position and scattered direction to result
        scattered_ray.origin = hit_record.p
        scattered_ray.direction = normalize(reflect(ray.direction, microfacet_normal_world))
        # transform scattered direction to tangent space
        sampled_vec_tangent = basis.transform_to(scattered_ray.direction)
        # check if we are reflecting off the top of the surface
        if(outgoing_vector_tangent[2] > 0 and sampled_vec_tangent[2] > 0):
            microfacet_normal_tangent = normalize(outgoing_vector_tangent + sampled_vec_tangent)
            # compute dot products for assisting BSDF evaluation
            n_dot_h = microfacet_normal_tangent[2]
            n_dot_v = outgoing_vector_tangent[2]
            n_dot_l = sampled_vec_tangent[2]
            h_dot_l = np.dot(microfacet_normal_tangent, sampled_vec_tangent)
            h_dot_v = np.dot(microfacet_normal_tangent, outgoing_vector_tangent)
            denom = (n_dot_h * n_dot_h) * (alpha2 - 1.0) + 1.0
            # computing microfacet normal distribution function
            D = alpha2 / (np.pi * denom * denom)
            k = alpha2 * 0.5
            # computing geometric shadowing function
            dv = n_dot_v / (n_dot_v*(1-k) + k)
            dl = n_dot_l / (n_dot_l*(1-k) + k)
            G = dv * dl
            # computing the importance sampling PDF of the microfacet distribution function
            pdf = 0.25 * D * n_dot_h / h_dot_l
            if(G > 0 and pdf > 0):
                # calculate reflectance
                bsdf = sampled_vec_tangent[2] * D * G / (4.0 * n_dot_l * n_dot_v)
                return (self.albedo * bsdf, pdf)
        # sampled ray not reflective, so no reflectance
        return (np.array([0.0, 0.0, 0.0]), 0.0)