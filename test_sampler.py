import numpy as np
import os
import random
import render as r

from sampler import *

if __name__ == '__main__':
    thread_count = 16
    sample_count = 4
    sample_type = 1
    max_bounce = 3
    r.render(thread_count, "snow_field_2_puresky_1k", sample_count, sample_type, True, max_bounce)