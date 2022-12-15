from enum import Enum
import json

""" FILES """
BSC5_path = '/home/ubuntu/StarTracker/BSC5'
BSC5ra_path = '/home/ubuntu/StarTracker/BSC5ra'

""" ENUMS """
class Epoch(Enum):
    J1950 = 1
    J2000 = 2

""" MISC CONSTANTS """
BYTEORDER = 'little'

""" CAMERA CONFIGS """
DEFAULT_ALVIUM = json.load(open("camera_configs/simulated_alvium.json"))