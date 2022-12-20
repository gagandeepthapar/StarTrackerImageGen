from io import BufferedReader
from typing import overload
import pandas as pd
import numpy as np
from .readerClass import GenericReader
import struct

class GenericStar(GenericReader):

    def __init__(self, object_size:int, object_lens:dict):
        self.object_size = object_size
        self.object_lens = object_lens
        super().__init__(object_size=self.object_size, object_lens=self.object_lens)

        self.object_params = {}
        return

    def read_star(self, catalog_file_pointer:BufferedReader) -> None:
        super().read_object(star_catalog_file=catalog_file_pointer)
        return

    def get_right_ascension_declination(self) -> None:
        return


class YaleStar(GenericStar):
    
    star_size = 32
    star_lens = {'catalog_number': 4,
                 'right_ascension': 8,
                 'declination': 8,
                 'spectral_type_a': 1,
                 'spectral_type_b': 1,
                 'v_magnitude': 2,
                 'ascension_proper_motion': 4,
                 'declination_proper_motion': 4}

    param_types = {'catalog_number': 'f',
                    'right_ascension': 'd',     # [rad]
                    'declination': 'd',         # [rad]
                    'spectral_type_a': 'c',
                    'spectral_type_b': 'c',
                    'v_magnitude': 'h',
                    'ascension_proper_motion': 'f',
                    'declination_proper_motion': 'f'} 
        

    def __init__(self):
        super().__init__(object_size=self.star_size, object_lens=self.star_lens)
        self.set_item_read_range(self.star_lens)

    def read_star(self, catalog_file_pointer:BufferedReader) -> None:

        object_data = catalog_file_pointer.read(self.star_size)
        
        for column in self.object_frame.index.tolist():
            start_read = self.object_frame.at[column, 'range'][0]
            stop_read = self.object_frame.at[column, 'range'][1]

            self.object_params[column] = [struct.unpack(self.param_types[column], object_data[start_read:stop_read])[0]]

        self.object_params['v_magnitude'][0]/=100
        
        return self.object_params
    
    def get_right_ascension_declination(self) -> float:
        if len(self.object_params == 0):
            return None, None
        
        return self.object_params['right_ascension'], self.object_params['declination']