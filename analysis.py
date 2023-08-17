import argparse
import imageio as iio
import matplotlib.pyplot as plt
import numpy as np
import os
import png
import random

from color import *

def compare_images(reference, result):
    # we assuming resolution fixed to 400x300
    width = 400
    height = 300
    total = 0.0
    for x in range(0, width):
        for y in range(0, height):
            diff = np.log(luminance(srgb_to_linear(reference[y][x]))+1) - np.log(luminance(srgb_to_linear(result[y][x]))+1)
            diff2 = diff * diff
            total = total + diff2
    return total / (width * height)

def image_comparison(lighting_file, image_count):
    algorithm = ["IS", "MLT_IS"]
    reference_file = "./results/" + lighting_file + "_is/is_1000_spp.png" 
    reference_image = iio.imread(reference_file)
    print("Reference Image Loaded. Image Detail (Height, Width, Channel): ", reference_image.shape)
    folder_path = "./results/" + lighting_file
    xaxis = np.arange(1, image_count+1)
    yaxes = []
    current_idx = 0
    for i in algorithm:
        compare_folder_path = folder_path
        if i == "IS":
            compare_folder_path = compare_folder_path + "_is/"
        elif i == "SIR":
            compare_folder_path = compare_folder_path + "_sir/"
        elif i == "MLT_IS":
            compare_folder_path = compare_folder_path + "_mlt_is/"
        else:
            msg = "Incorrect algorithm type: " + i
            print(msg)
            return
        mses = np.zeros(image_count)
        for image_idx in range(0, image_count):
            print("Format: %d/%d Image: %d/%d" % (current_idx, len(algorithm), image_idx, image_count))
            compare_image_path = compare_folder_path
            if i == "IS":
                compare_image_path = compare_image_path + "is_" + str(image_idx+1) + "_spp.png"
            elif i == "SIR":
                compare_image_path = compare_image_path + "sir_" + str(image_idx+1) + "_spp.png"
            elif i == "MLT_IS":
                compare_image_path = compare_image_path + "is_" + str(image_idx+1) + "_spp.png"
            result_image = iio.imread(compare_image_path)
            mse = compare_images(reference_image, result_image)
            mses[image_idx]=mse
        yaxes.append(mses)
        current_idx = current_idx + 1
    fig, ax = plt.subplots()
    ax.plot(xaxis, yaxes[0])
    ax.plot(xaxis, yaxes[1])
    ax.set_title(lighting_file)
    ax.legend(algorithm)
    ax.set_xlabel("Sample count")
    ax.set_ylabel("log(MSE)")
    plt.savefig("./results/" + lighting_file + "_" + str(image_count) + "_" + algorithm[0] + "_" + algorithm[1] + "_samples.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lighting", 
        type=str, 
        required=True,
        help="Lighting file name, i.e. snow_field_2_puresky_1k"
    )
    parser.add_argument(
        '--image_count',
        type=int,
        required=True,
        help="No. of images to compare with reference"
    )
    args = parser.parse_args()
    image_comparison(args.lighting, args.image_count)