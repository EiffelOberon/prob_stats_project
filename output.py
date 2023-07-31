import argparse
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
from sphere import *
from trace import *

def path_trace_row(frame, progress, start_y, step, width, height, scene, paths, sample_idx, samples):
    for y in range(start_y, height, step):
        if(y >= height):
            return
        for x in range(width):
            color = np.array([0.0, 0.0, 0.0])
            path_index = y * width + x
            paths[path_index] = Path(x, y, path_index)
            paths[path_index].bounce = 0
            # calculate screen [0,1] coordinate
            u = float(x + random.random())/float(width)
            v = float((height - y) + random.random())/float(height)
            cam_ray = scene.camera.get_ray(u, v)
            # start tracing rays
            color = color + trace(scene, cam_ray, paths[path_index])
            frame.accumulation[y * width + x] = frame.accumulation[y * width + x] + color
            color = linear_to_srgb(frame.accumulation[y * width + x] / (sample_idx + 1))
            # set image
            frame.display_img[y][x] = np.array([color[0], color[1], color[2]])
        progress[y] = 1
        print("Progress: %.2f%% [Sample: %i/%i]" % ((np.clip((np.sum(progress) / height), 0.0, 1.0) * 100), (sample_idx + 1), samples))

def threaded_path_trace(frame, sky, width, height, scene, samples, threads):
    step = threads
    threads = [None] * step
    paths = [None] * width * height
    # record start time
    start = time.perf_counter()
    # figure for displaying image during render
    plt.ion()
    fig, ax = plt.subplots()
    for sample_idx in range(0, samples, 1):
        progress = [0] * height
        scene.init_random(sample_idx)
        # start threads
        for i in range(step):
            threads[i] = threading.Thread(target=path_trace_row, args=(frame, progress, i, step, width, height, scene, paths, sample_idx, samples))
            threads[i].start()
        # display image interactively
        while(np.sum(progress) < height):
            image_display = ax.imshow(frame.display_img, extent=[0, width, 0, height])
            fig.canvas.flush_events()
            plt.show()
            time.sleep(0.5)
        # wait for all threads to finish
        for i in range(step):
            threads[i].join()
        # copy results
        for y in range(0, height):
            rgb_row = ()
            for x in range(width):
                color = frame.accumulation[y * width + x] / (sample_idx + 1)
                color = linear_to_srgb(color)
                rgb_row = rgb_row + (color[0], color[1], color[2])
            frame.img[y] = rgb_row
        # save image
        folder = "./results/" + sky + "_{}/"
        file = "./results/" + sky + "_{}/{}_{}_spp.png"
        if(scene.sampling == Sampling.IMPORTANCE_SAMPLING):
            folder = folder.format("is")
            file = file.format("is", "is", (sample_idx + 1))
        else:
            folder = folder.format("sir")
            file = file.format("sir", "sir", (sample_idx + 1))

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

    # display finished image
    image_display = ax.imshow(frame.display_img, extent=[0, width, 0, height])
    fig.canvas.flush_events()
    plt.show()
    plt.show(block=True)
    print("Finished rendering - saving image")

def threaded_mlt(frame, sky, width, height, scene, samples, threads):
    step = threads
    threads = [None] * step
    paths = [None] * width * height
    # record start time
    start = time.perf_counter()
    # figure for displaying image during render
    plt.ion()
    fig, ax = plt.subplots()
    sample_idx = 0
    progress = [0] * height
    scene.init_random(sample_idx)
    # start threads
    for i in range(step):
        threads[i] = threading.Thread(target=path_trace_row, args=(frame, progress, i, step, width, height, scene, paths, sample_idx, samples))
        threads[i].start()
    # display image interactively
    while(np.sum(progress) < height):
        image_display = ax.imshow(frame.display_img, extent=[0, width, 0, height])
        fig.canvas.flush_events()
        plt.show()
        time.sleep(0.5)
    # wait for all threads to finish
    for i in range(step):
        threads[i].join()
    # copy results
    for y in range(0, height):
        rgb_row = ()
        for x in range(width):
            color = frame.accumulation[y * width + x] / (sample_idx + 1)
            color = linear_to_srgb(color)
            rgb_row = rgb_row + (color[0], color[1], color[2])
        frame.img[y] = rgb_row
    # save image
    folder = "./results/" + sky + "_{}/"
    file = "./results/" + sky + "_{}/{}_{}_spp.png"
    if(scene.sampling == Sampling.IMPORTANCE_SAMPLING):
        folder = folder.format("mlt_is")
        file = file.format("mlt_is", "is", (sample_idx + 1))
    else:
        folder = folder.format("mlt_sir")
        file = file.format("mlt_sir", "sir", (sample_idx + 1))

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

    # display finished image
    image_display = ax.imshow(frame.display_img, extent=[0, width, 0, height])
    fig.canvas.flush_events()
    plt.show()
    plt.show(block=True)
    print("Finished rendering - saving image")

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
    # initialize image
    frame = Frame(width, height)
    # path trace
    if(mlt):
        threaded_mlt(frame, sky, width, height, scene, samples, threads)
    else:
        threaded_path_trace(frame, sky, width, height, scene, samples, threads)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--thread_count',
        type=int,
        nargs='?',
        default=8,
        help="No. of threads for multi-threaded rendering. (Default: 8)"
    )
    parser.add_argument(
        '--sample_count',
        type=int,
        nargs='?',
        default=1,
        help="No. of samples per pixel. (Default: 1)"
    )
    parser.add_argument(
        '--sample_type',
        type=int,
        nargs='?',
        default=8,
        help="Sample type, 1: importance sampling, 2: sampling importance resampling. (Default: 1)"
    )
    parser.add_argument(
        '--max_bounce',
        type=int,
        nargs='?',
        default=2,
        help="Max number of bounce. (Default: 2)"
    )
    parser.add_argument(
        "--sky", 
        type=str, 
        required=True,
        help="Environment map file name, i.e. snow_field_2_puresky_1k"
    )
    parser.add_argument(
        '--mlt',
        action='store_true'
    )
    args = parser.parse_args()
    render(args.thread_count, args.sky, args.sample_count, args.sample_type, args.mlt, args.max_bounce)