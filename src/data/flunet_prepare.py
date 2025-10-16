import re # regular expression to find & replace text 
from pathlib import Path
from tkinter import N # pathlib to handle file paths 
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

def pick_column(df: pd.DataFrame, keywords: list[str]) -> str | None:
    """Pick columns from a DataFrame based on keywords."""
    lower_map = {c.lower(): c for c in df.columns}

    for k in lower_map:
        if all(word in k for word in keywords):
            return lower_map[k]
        return None

def get_country_column(df: pd.DataFrame) -> str:
    """Get the country column from a DataFrame."""
    for keys in [['Country area or territory'], ['country']]:
        col = pick_column(df, keys)
        if col:
            return col
        raise ValueError('No country column found!')

def get_week_start_column(df: pd.DataFrame) -> str | None:
    for keys in [['Week start date (ISO 8601 calendar)'], ['week_start']]:
        col = pick_column(df, keys)
        if col:
            return col
        return None

def get_year_week_column(df: pd.DataFrame) -> tuple[str, str] | tuple[None, None]:
    year_col = pick_column(df, ['year'])
    week_col = pick_column(df, ['week'])

    if year_col and week_col:
        return year_col, week_col
    
    return None, None

def to_week_start_date(df: pd.DataFrame) -> pd.Series:
    """ prefer explicit week start date if present """ 
    date_col = get_week_start_column(df)
    if date_col:
        return pd.to_datetime(df[date_col], errors='coerce')

    """ otherwise, calculate from year and week """
    year_col, week_col = get_year_week_column(df)
    if not year_col or not week_col:
        raise ValueError('Missing year or week column!')

    year = df[year_col].astype(str).str.extract(r"(\d{4})", expand = False) 
    week = df[week_col].astype(str).str.extract(r"(\d{1,2})", expand = False) 

    as_str = year + week + '1' # monday = 1
    return pd.to_datetime(as_str, format = '%G%V%u', errors = 'coerce')
    
      