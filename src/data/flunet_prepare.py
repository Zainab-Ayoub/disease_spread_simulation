import re # regular expression to find & replace text 
from pathlib import Path # pathlib to handle file paths 
import numpy as np 
import pandas as pd

try:
    import pyarrow
    HAS_PARQUET = True
except:
    HAS_PARQUEST = False 

raw_dir = Path('data/raw/flunet_weekly.csv')
out_dir = Path('data/processed')
out_dir.mkdir(parents=True, exist_ok = True)

