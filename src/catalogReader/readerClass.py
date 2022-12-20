from io import BufferedReader
from re import I
import sys, os
import struct

sys.path.append('../../')

import pandas as pd

""" 
This file contains the class to read an arbitrary object

Does:
    Read object

Does Not Do:
    Specify catalog/star specific information

"""

class GenericReader:
    """ Generic class to be inherited by different catalogs
    """    
    
    def __init__(self, object_size:int, object_lens:dict):

        self.object_size = object_size
        self.object_items = list(object_lens.keys())
        self.object_frame = pd.DataFrame(columns=['range', 'value'], index=self.object_items)

        self.set_item_read_range(object_lens)

    def __eq__(self, other):
        if isinstance(other, GenericReader):
            return self.object_frame.eq(other.object_frame)
        return False


    def read_object(self, star_catalog_file:BufferedReader) -> None:

        """
        reads object from, sets the values of objects and moves file pointer to right after object

        args:
            star_catalog_file: str: string of path to file of star catalog to read

        returns:
            None, updates self.object_params

        raises:
            None
        """

        object_data = star_catalog_file.read(self.object_size)        

        for column in self.object_items:
            start_read = self.object_frame.at[column, 'range'][0]
            stop_read = self.object_frame.at[column, 'range'][1]
            self.object_frame.at[column, 'value'] = struct.unpack('i', object_data[start_read:stop_read])[0]
        
        return


    def set_item_read_range(self, object_lens) -> None:
        """
        determines breakup of object based on length of column values

        args:
            None; self.object_lens
        
        returns:
            None; updates self.heade_range
        
        raises:
            None
        """

        range_start = 0

        for x in object_lens:
            final = object_lens[x] + range_start
            self.object_frame.at[x, 'range'] = [range_start, final]
            range_start += object_lens[x]

        return

