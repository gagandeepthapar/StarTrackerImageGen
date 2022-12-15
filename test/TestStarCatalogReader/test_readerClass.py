import sys
sys.path.append('../../')
import pandas as pd
import pytest
import logging

from src.StarCatalogReader.readerClass import *

class TestGenericReader:

    size = 12
    headers = {'idxA': 4, 'idxB': 4, 'idxC': 4}

    def test_init(self):
        reader = GenericReader(object_size=self.size,
                               object_lens=self.headers)

        expected = pd.DataFrame(index=['idxA', 'idxB', 'idxC'],
                                columns=['range', 'value'])
        
        expected.at['idxA', 'range'] = [0,4]
        expected.at['idxB', 'range'] = [4, 8]
        expected.at['idxC', 'range'] = [8,12]

        print(reader.object_frame)
        print(expected)

        assert reader.object_frame.equals(expected)
