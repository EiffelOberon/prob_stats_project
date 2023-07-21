class Path:
    def __init__(self, x, y, index):
        # the image coordinate of the path
        self.x = x
        self.y = y
        self.index = index
        self.bounce = 0

    