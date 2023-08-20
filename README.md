# 625.721 SU23 Probability and Stochastic Process I - Bayesian Sampling for Light Transport Simulation 

Final Project By Wallace Yuen

## About

This project contains code for rendering 3D graphical images with a simple scene inspired by Ray Tracing in one weekend, by Peter Shirley, et al. The code base is in python and would require several python modules to execute, which are listed below. The scene rendered here is extremely simple as the purpose of the project is to compare different sampling algorithms. This application utilizes environment HDRI as light source for the scene, and the HDRI files are obtained via [Polyhaven](https://polyhaven.com/).

The renderer currently runs on MacOS and Windows, but only thoroughly tested with Windows. The renderer comes with command-line interface only, and renders with diffuse BRDF, GGX with importance sampling and also sampling importance resampling. Currently, the renderer renders with either Monte Carlo path tracing or Metropolis light transport (primary space Metropolis light transport to be more accurate).

## How to run

In order to run the python script, the following libraries have to be installed for python, mainly for the loading of HDRI, saving and loading image files.

```
pip install imageio
pip install pypng
pip install opencv-python
```

If you have not already installed the typical math modules, you would need:

```
pip install matplotlib
```

An example of how to render an image with Monte Carlo path tracing:
```
python ./output.py --sky=farm_field_puresky_1k --sample_type=1 --sample_count=100
```

All the possible options are documented in:
```
python ./output.py -h
```

The list of images for environmment background for the option i.e., `--sky="belfast_sunset_puresky_1k"` available (credits to Polyhaven):
```
belfast_sunset_puresky_1k
drakensberg_solitary_mountain_puresky_1k
farm_field_puresky_1k
kloofendal_28d_misty_puresky_1k
kloofendal_48d_partly_cloudy_puresky_1k
kloppenheim_02_puresky_1k
kloppenheim_05_puresky_1k
kloppenheim_06_puresky_1k
snow_field_2_puresky_1k
sunflowers_puresky_1k
syferfontein_0d_clear_puresky_1k
table_mountain_2_puresky_1k
wasteland_clouds_puresky_1k
```