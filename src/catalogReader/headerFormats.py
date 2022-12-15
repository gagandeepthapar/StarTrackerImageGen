from io import BufferedReader
import os, sys
sys.path.append('../../')
from .readerClass import GenericReader
import numpy as np

""" 
This file outlines different header formats for various star catalogs

Does:
    Read Header
    Create reference for different star properties
    Set global references i.e., J2000 EPOCH

Does Not Do:
    Read star info
    Create information

"""

class GenericHeader(GenericReader):

    def __init__(self, object_size:int, object_lens:dict) -> None:
        super().__init__(object_size=object_size, object_lens=object_lens)
        return

    def read_header(self, catalog_file_pointer:BufferedReader) -> None:
        super().read_object(star_catalog_file=catalog_file_pointer)
        return

class YaleHeader(GenericHeader):
    """ Yale Bright Star Catalog (BSC5/ra) specific Header Function
    """

    header_size = 28
    header_lens = {'sequence_number': 4,
                   'first_star_number': 4,
                   'number_of_stars': 4,
                   'star_number_present_flag': 4,
                   'proper_motion_present_flag': 4,
                   'number_of_magnitudes': 4,
                   'number_of_bytes_per_star': 4}

    def __init__(self):
        super().__init__(object_size=self.header_size, object_lens=self.header_lens)

    def read_header(self, catalog_file_pointer:BufferedReader):
        super().read_header(catalog_file_pointer=catalog_file_pointer)
        self.num_stars = np.abs(self.object_frame.at['number_of_stars', 'value'])

