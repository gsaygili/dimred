import pandas as pd
from pathlib import Path
import numpy as np
from sys import platform


if platform == "linux" or platform == "linux2":
    data_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/"
elif platform == "darwin":
    data_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/"
elif platform == "win32":
    data_folder = "C:/Users/Admin/Documents/GitHub/dimred/datasets/Ovarian-PBSII-061902/"

data = pd.read_csv(data_folder+"ovarian-cancer-nci-pbsii-data-no-header.csv").to_numpy()
np.save(data_folder+'emb_p30/' + "X", data)
