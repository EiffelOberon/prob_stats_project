from enum import Enum
from enum import IntEnum

class Sampling(Enum):
    IMPORTANCE_SAMPLING = 1
    IMPORTANCE_RESAMPLING = 2

class RandomNumber(IntEnum):
    RANDOM_BRDF_U = 0
    RANDOM_BRDF_V = 1
    RANDOM_RIS_1 = 2
    RANDOM_RIS_2 = 3
    RANDOM_RIS_3 = 4
    RANDOM_RIS_4 = 5
    RANDOM_RIS_5 = 6
    RANDOM_RIS_6 = 7
    RANDOM_RIS_7 = 8
    RANDOM_RIS_8 = 9
    RANDOM_COUNT = 10