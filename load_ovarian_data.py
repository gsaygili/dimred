import pandas as pd
from pathlib import Path
import numpy as np
from sys import platform


if platform == "linux" or platform == "linux2":
    data_folder = "/home/gorkem/datasets/"
elif platform == "darwin":
    data_folder = "/home/gorkem/datasets/"
elif platform == "win32":
    data_folder = "C:/Users/gsayg/Dropbox/datasets/ovarian-cancer-nci-pbsii-data/"

data = pd.read_csv(data_folder+"ovarian-cancer-nci-pbsii-data-no-header.csv").to_numpy()

