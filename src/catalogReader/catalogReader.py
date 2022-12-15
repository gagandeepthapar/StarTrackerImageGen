import pandas as pd
import numpy as np
from .headerFormats import GenericHeader
from .star import GenericStar 

from .starlist import StarList

class CatalogReader:

    def __init__(self, path_to_file:str, header_type:GenericHeader, star_type:GenericStar):
        self.path_to_file = path_to_file
        self.header = header_type
        self.star = star_type
        self.starlist = StarList(self.star.object_lens.keys())

    def read_file(self) -> None:
        fp = open(self.path_to_file, 'rb')
        self.header.read_header(catalog_file_pointer=fp)
        self.starlist.num_stars = self.header.num_stars
        
        for i in range(int(self.starlist.num_stars)):
            star_series = pd.DataFrame(self.star.read_star(fp))
            self.starlist.add_to_frame(star_series=star_series)
            # print(f'{i}/{int(self.starlist.num_stars/10)}')

        fp.close()

        return self.starlist

