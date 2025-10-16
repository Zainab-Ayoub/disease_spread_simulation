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

def read_csv(csv_path: Path) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame."""
    if not csv_path.exists():
        raise FileNotFoundError(f'Raw file not found: {csv_path.resolve()}')
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df

