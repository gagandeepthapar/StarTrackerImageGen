from enum import Enum
import json
import os

from .pickleCatalogs import convCatalogPickle as ccp

""" CURRENT FILE """
curFile = os.path.abspath(os.path.dirname(__file__))

""" FILES """
BSC5_path = os.path.join(curFile, 'BSC5')
BSC5ra_path = '/home/ubuntu/StarTracker/BSC5ra'

""" ENUMS """
class Epoch(Enum):
    J1950 = 1
    J2000 = 2

""" MISC CONSTANTS """
BYTEORDER = 'little'

""" CAMERA CONFIGS """
DEFAULT_ALVIUM = os.path.join(curFile, 'cameraConfigs/', 'simulated_alvium.json')

""" CATALOGS """
YBSC_PATH = os.path.join(curFile, 'pickleCatalogs/', 'YBSC.pkl')
ybsc_exist = os.path.exists(YBSC_PATH)

if not ybsc_exist:
    print('PKL File Does Not Exist; Creating Now...\n')
    ybsc_frame = ccp.readCatalogFromFile(BSC5_path)
    ccp.convCatalog(ybsc_frame, YBSC_PATH)

""" IMAGE DIRECTORIES """
SIM_IMAGES = os.path.join(curFile,'_StarTrackerTestImages/','simImages/')
REAL_IMAGES = os.path.join(curFile, '_StarTrackerTestImages/','nightSkyImages/')