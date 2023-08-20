import imageio as iio
import numpy as np
import os
import png
import random
import threading
import time

import matplotlib 
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

from datetime import timedelta

from camera import *
from color import *
from frame import Frame
from path import Path
from ray import Ray
from sampler import *
from sphere import *
from trace import *

def radiance(frame, sampler, scene, bootstrap):
    if(bootstrap == False):
        # prepare sampler for this iteration if MCMC
        sampler.start_iteration()
    # get next samples for camera rays
    u = sampler.next()
    v = sampler.next()
    # get camera ray
    cam_ray = scene.camera.get_ray(u, v)
    # compute screen coordinates
    x = min(int(u * sampler.width), sampler.width - 1)
    y = min(int(sampler.height - v * sampler.height), sampler.height - 1)
    # record radiance for this pixel
    r = RadianceRecord(x, y)
    # start tracing rays
    path = Path(x, y)
    r.radiance = trace(scene, sampler, cam_ray, path)
    return r

def path_trace_row(frame, progress, start_y, step, width, height, scene, paths, sample_idx, samples):
    seed = random.randint(0, 4294967296)
    sampler = Sampler(width, height, seed)
    for y in range(start_y, height, step):
        if(y >= height):
            return
        for x in range(width):
            color = np.array([0.0, 0.0, 0.0])
            path_index = y * width + x
            paths[path_index] = Path(x, y)
            # calculate screen [0,1] coordinate
            u = float(x + sampler.next())/float(width)
            v = float((height - y) + sampler.next())/float(height)
            cam_ray = scene.camera.get_ray(u, v)
            # start tracing rays
            color = color + trace(scene, sampler, cam_ray, paths[path_index])
            frame.accumulation[y * width + x] = frame.accumulation[y * width + x] + color
        progress[y] = 1
        print("Progress: %.2f%% [Sample: %i/%i]" % ((np.clip((np.sum(progress) / height), 0.0, 1.0) * 100), (sample_idx + 1), samples))

def threaded_path_trace(frame, sky, width, height, scene, samples, threads):
    step = threads
    threads = [None] * step
    paths = [None] * width * height
    # record start time
    start = time.perf_counter()
    for sample_idx in range(0, samples, 1):
        progress = [0] * height
        # start threads
        for i in range(step):
            threads[i] = threading.Thread(target=path_trace_row, args=(frame, progress, i, step, width, height, scene, paths, sample_idx, samples))
            threads[i].start()
        # wait for all threads to finish
        for i in range(step):
            threads[i].join()
    # copy results
    for y in range(0, height):
        rgb_row = ()
        for x in range(width):
            color = frame.accumulation[y * width + x] / (samples)
            color = linear_to_srgb(color)
            rgb_row = rgb_row + (color[0], color[1], color[2])
        frame.img[y] = rgb_row
    # save image
    folder = "./results/" + sky + "_{}/"
    file = "./results/" + sky + "_{}/{}_{}_spp.png"
    if(scene.sampling == Sampling.IMPORTANCE_SAMPLING):
        folder = folder.format("is")
        file = file.format("is", "is", (samples))
    else:
        folder = folder.format("sir")
        file = file.format("sir", "sir", (samples))

    isExist = os.path.exists(folder)
    if isExist == False:
        os.makedirs(folder)
    with open(file, 'wb') as f:
        w = png.Writer(width, height, greyscale=False, gamma=1.0)
        w.write(f, frame.img)
        print("Image saved")
    # record end time
    end = time.perf_counter()
    duration = timedelta(seconds=end-start)
    print("Render completed in: ", duration)

def run_mcmc(frame, sampler, scene, b):
    radiance_record = radiance(frame, sampler, scene, False)
    proposed_luminance = luminance(radiance_record.radiance)
    current_luminance = luminance(sampler.current_record.radiance)
    # compute acceptance ratio based on luminance
    ratio = 0.0
    if(current_luminance > 0.0):
        ratio = proposed_luminance / current_luminance
    acceptance_ratio = max(0.0, min(1.0, ratio))
    additional_weight = 0.0
    if(sampler.large_step):
        additional_weight = 1.0
    # two weights for two results, PSSMLT does not discard the rejected samples completely
    weight1 = (acceptance_ratio + additional_weight) / (proposed_luminance / b + 0.25)
    weight2 = (1.0 - acceptance_ratio) / (current_luminance / b + 0.25)
    # two results
    result1 = RadianceRecord(0, 0)
    result1.x = radiance_record.x
    result1.y = radiance_record.y
    result1.radiance = radiance_record.radiance * weight1
    result2 = RadianceRecord(0, 0)
    result2.x = sampler.current_record.x
    result2.y = sampler.current_record.y
    result2.radiance = sampler.current_record.radiance * weight2
    # determine if we accept
    if(acceptance_ratio == 1.0 or sampler.uniform() < acceptance_ratio):
        sampler.accept()
        sampler.current_record = radiance_record
    else:
        sampler.reject()
    return [result1, result2]


def threaded_mlt(frame, sky, width, height, scene, samples, threads):
    step = threads
    threads = [None] * step
    paths = [None] * width * height
    # record start time
    start = time.perf_counter()
    # set fixed seed for bootstrap
    random.seed(1)
    bootstrap_count = 100000
    # prepare arrays
    seeds = np.zeros(bootstrap_count)
    weights = np.zeros(bootstrap_count)
    cdf = np.zeros(bootstrap_count)
    for i in range(bootstrap_count):
        seeds[i] = random.randint(0, 4294967296)
    weight_sum = 0.0
    for i in range(bootstrap_count):
        sampler = RandomNumberSampler(width, height, seeds[i])
        r = radiance(frame, sampler, scene, True)
        weights[i] = luminance(r.radiance)
        weight_sum += weights[i]
    cdf[0] = weights[0]
    for i in range(1, bootstrap_count):
        cdf[i] = cdf[i-1] + weights[i]
    # last cdf over number of bootstrap
    b = cdf[-1] / bootstrap_count
    # start mcmc process now
    count = 0
    chains_count = 4096
    mutations_count = int(np.ceil(float(width) * height * samples / chains_count))
    print("chain count: %d, mutation count:%d" % (chains_count, mutations_count))
    for i in range(0, chains_count):
        # search for the seed that is the path we want to evaluate from the
        # cdf list we have accumulated
        r = random.random() * cdf[-1]
        k = 1
        while(k < bootstrap_count):
            if(cdf[k-1] < r and r <= cdf[k]):
                break
            k += 1
        k -= 1
        # found seed, make sampler
        sampler = MetropolisSampler(width, height, seeds[k])
        sampler.current_record = radiance(frame, sampler, scene, False)
        # reseed after first record for mutation
        sampler.seed = random.randint(0, 4294967296)
        for i in range(0, mutations_count):
            results = run_mcmc(frame, sampler, scene, b)
            # first result splat
            y = results[0].y
            x = results[0].x
            frame.accumulation[y * width + x] = frame.accumulation[y * width + x] + results[0].radiance
            # second result splat
            y = results[1].y
            x = results[1].x
            frame.accumulation[y * width + x] = frame.accumulation[y * width + x] + results[1].radiance
        count += 1
        print("Done Markov Chain %d/%d acceptance rate: %f" % (count, chains_count, float(sampler.accepted) / float(sampler.accepted + sampler.rejected)))
   
    for y in range(0, height):
        rgb_row = ()
        for x in range(width):
            color = linear_to_srgb(frame.accumulation[y * width + x] / (samples))
            rgb_row = rgb_row + (color[0], color[1], color[2])
        frame.img[y]=rgb_row

    # save image
    folder = "./results/" + sky + "_{}/"
    file = "./results/" + sky + "_{}/{}_{}_spp.png"
    if(scene.sampling == Sampling.IMPORTANCE_SAMPLING):
        folder = folder.format("mlt_is")
        file = file.format("mlt_is", "is", (samples))
    else:
        folder = folder.format("mlt_sir")
        file = file.format("mlt_sir", "sir", (samples))

    isExist = os.path.exists(folder)
    if isExist == False:
        os.makedirs(folder)
    with open(file, 'wb') as f:
        w = png.Writer(width, height, greyscale=False, gamma=1.0)
        w.write(f, frame.img)
        print("Image saved")
    
    # record end time
    end = time.perf_counter()
    duration = timedelta(seconds=end-start)
    print("Render completed in: ", duration)

def render(threads, sky, samples, sample_type, mlt, max_bounce):
    # read sky
    print("Loading sky image")
    # required for processing HDR images properly
    iio.plugins.freeimage.download()
    sky_file_name = "./"+ sky +".hdr"
    sky_image = iio.imread(sky_file_name)
    
    # the HDRI loaded in macOS is [0, 255] not [0.0, 1.0], so
    # we normalize it for just macOS
    if platform == "darwin":
        sky_image = sky_image * 1.0 / 255.0
    print("Image Loaded. Image Detail (Height, Width, Channel): ", sky_image.shape)

    print("Begin rendering")
    width = 400
    height = 300
    
    sample_algorithm = Sampling.IMPORTANCE_SAMPLING
    if(sample_type == 2):
        sample_algorithm = Sampling.IMPORTANCE_RESAMPLING

    # initialize scene
    scene = Scene(sample_algorithm, width, height, sky_image, max_bounce)
    # path trace
    if(mlt):
        # initialize image
        frame = Frame(width, height)
        threaded_mlt(frame, sky, width, height, scene, samples, threads)
    else:
        # initialize image
        frame = Frame(width, height)
        threaded_path_trace(frame, sky, width, height, scene, samples, threads)
    