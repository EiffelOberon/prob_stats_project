import numpy as np

class Frame:
    def __init__(self, width, height):
        # initialize image
        self.img = []
        self.accumulation = []
        for y in range(height):
            rgb_row = ()
            for x in range(width):
                rgb_row = rgb_row + (1, 1, 1)
                self.accumulation.append(np.array([0, 0, 0]))
            self.img.append(rgb_row)
