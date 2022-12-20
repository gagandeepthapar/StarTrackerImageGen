from ..src.catalogReader.catalogReader import CatalogReader
from ..src.catalogReader.headerFormats import YaleHeader
from ..src.catalogReader.star import YaleStar

import pandas as pd

import argparse
# from constants import BSC5_path

def readCatalogFromFile(cat_fp:str)->pd.DataFrame:

    reader = CatalogReader(cat_fp, YaleHeader(), YaleStar())
    reader.read_file()
    starframe = reader.starlist.star_frame

    return starframe

def convCatalog(starframe:pd.DataFrame, pkl_fp:str)->None:

    starframe.to_pickle(pkl_fp)

    return

def parseArguments()->argparse.Namespace:

    parser = argparse.ArgumentParser(description='Set catalog to read and convert to pkl file')

    parser.add_argument('-cfp',
                            help='Set filepath for catalog file; Default: BSC5')

    parser.add_argument('-pfp',
                            help='Set default for output pickle; Default: ./catalogPickle.pkl',
                            default='catalogPickle.pkl')

    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parseArguments()
    fullCatalog = readCatalogFromFile(args.cfp)
    convCatalog(fullCatalog, args.pfp)