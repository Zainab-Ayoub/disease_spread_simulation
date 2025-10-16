import re # regular expression to find & replace text 
from pathlib import Path
from tkinter import N # pathlib to handle file paths 
import numpy as np 
import pandas as pd

try:
    import pyarrow
    HAS_PARQUET = True
except:
    HAS_PARQUET = False 

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

def find_positive_columns(df: pd.DataFrame) -> list[str]:
    cols_lower = {c.lower(): c for c in df.columns}  

    """ if there's a single 'positive' total column, use it """
    for key in ['influenza positive', 'total positive', 'positives', 'positive']:
        for k in cols_lower:
            if key in k and '%' not in k and 'percent' not in k:
                return [cols_lower[k]]

    """ else sum across subtype columns """
    tokens = [
        'h1', 'pdm09', 'h3', 'h5', 'not subtyped', 'victoria', 'yamagata', 'lineage not deterined'
    ]    
    matches = []
    for k in cols_lower:
        if any(tok in k for tok in tokens):
            matches.append(cols_lower[k])
    return matches

def clean(df: pd.DataFrame) -> pd.DataFrame:
    country_col = get_country_column(df)
    week_start = to_week_start_date(df)
    pos_cols = find_positive_columns(df)

    if not pos_cols:
        raise ValueError('No positive columns found!')

    df_out = pd.DataFrame(
        {
            'country_name': df[country_col].astype(str).str.strip(),
            'week_start_date': week_start,
            'positives': df[pos_cols].apply(pd.to_numeric, errors = 'coerce').fillna(0).sum(axis = 1),
        }
    )

    df_out = df_out.dropna(subset=['week_start_date'])
    df_out['positives'] = df_out['positives'].astype(int)

    # tidy per country/week
    df_out = (
        df_out.group_by(['country_name', 'week_start_date'], as_index = False)['positives'].sum().sort_values(['country_name', 'week_start_date'])
    )

    return df_out

def add_splits(df: pd.DataFrame, val_weeks: int = 52) -> pd.DataFrame:
    """ last 'val_weeks' per country are for validation, rest is training """     
    df = df.copy()
    df['split'] = 'train'

    def mark_split(group: pd.DataFrame) -> pd.DataFrame:
        if len(group) == 0:
            return group
        idx = group.index[-val_weeks:] # tail by position 
        group.loc[idx, 'split'] = 'val'
        return group

    return df.groupby("country_name", group_keys = False).apply(mark_split)

def save_outputs(df: pd.DataFrame) -> None:
    out_csv = out_dir / 'flunet_weekly_clean.csv'
    df.to_csv(out_csv, index = False)

    if HAS_PARQUET:
        out_parq = out_dir / 'flunet_weekly.parquet'
        df.to_parquet(out_parq, index = False)

    print(f'Wrote: {out_csv}')

    if HAS_PARQUET:
        print(f'Wrote: {out_parq}')

def main() -> None:
    df_raw = read_raw(raw_dir)
    df_clean = clean(df_raw)
    df_split = add_splits(df_clean, val_weeks = 52)

    # quick summary
    num_countries = df_split['country_name'].nunique()
    date_min = df_split['week_start_date'].min()
    date_max = df_split['week_start_date'].max()

    print(f'Rows={len(df_split)}, Countries={num_countries}, Range={date_min.date()} → {date_max.date()}')
    save_outputs(df_split)

if __name__ == '__main__':
    main()    