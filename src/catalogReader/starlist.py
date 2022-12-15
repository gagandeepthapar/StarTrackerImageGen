import pandas as pd
import numpy as np

class StarList:

    num_stars: int
    star_frame: pd.DataFrame
    headers: list[str]

    def __init__(self, headers:list):
        self.headers = headers
        self.star_frame = pd.DataFrame(columns=self.headers)

    def add_to_frame(self, star_series:pd.Series):
        self.star_frame = pd.concat([self.star_frame, star_series])