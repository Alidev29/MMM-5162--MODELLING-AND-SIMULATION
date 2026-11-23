# File: weather_data.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Download (once) and read NASA POWER hourly wind speed and air temperature for the site;
#          daytime wind sample, monthly daytime temperature, and the selected wind distribution file.
# Inputs: site latitude/longitude, years, raw CSV path, selected-distribution CSV path
# Outputs: data/raw/nasa_power_madinah_2021_2023.csv; DataFrame[time, ws10m, t2m]; scipy frozen distribution
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import io
import os
import pandas as pd
import requests
from scipy import stats

POWER_URL = 'https://power.larc.nasa.gov/api/temporal/hourly/point'
FILL_VALUE = -999.0
REQUEST_TIMEOUT_S = 120
DAY_START_HOUR = 6
DAY_END_HOUR = 18


def _read_power_text(text):
    """Parse one POWER CSV text (header block between -BEGIN HEADER- and -END HEADER-)."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == '-END HEADER-') + 1 \
        if any(line.strip() == '-END HEADER-' for line in lines) else 0
    return pd.read_csv(io.StringIO('\n'.join(lines[start:])))


def download_power_hourly(lat, lon, start_year, end_year, out_csv):
    """Download POWER hourly WS10M and T2M (LST) year by year and merge into out_csv; skip if it exists."""
    if os.path.exists(out_csv):
        return out_csv
    frames = []
    for year in range(start_year, end_year + 1):
        query = {'parameters': 'WS10M,T2M', 'community': 'RE', 'format': 'CSV', 'time-standard': 'LST',
                 'latitude': lat, 'longitude': lon, 'start': f'{year}0101', 'end': f'{year}1231'}
        response = requests.get(POWER_URL, params=query, timeout=REQUEST_TIMEOUT_S)
        response.raise_for_status()
        frames.append(_read_power_text(response.text))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
    return out_csv


def load_power_hourly(csv_path):
    """Read a POWER hourly CSV (with or without its header block); drop -999 fill rows."""
    with open(csv_path, encoding='utf-8') as handle:
        df_raw = _read_power_text(handle.read())
    df_raw = df_raw[(df_raw['WS10M'] > FILL_VALUE + 1) & (df_raw['T2M'] > FILL_VALUE + 1)]
    time = pd.to_datetime(dict(year=df_raw['YEAR'], month=df_raw['MO'], day=df_raw['DY'], hour=df_raw['HR']))
    return pd.DataFrame({'time': time.to_numpy(), 'ws10m': df_raw['WS10M'].to_numpy(dtype=float),
                         't2m': df_raw['T2M'].to_numpy(dtype=float)})


def _daytime(df, start_hour, end_hour):
    hours = df['time'].dt.hour
    return df[(hours >= start_hour) & (hours <= end_hour)]


def daytime_wind(df, start_hour=DAY_START_HOUR, end_hour=DAY_END_HOUR):
    """Hourly wind speeds between start_hour and end_hour (inclusive), strictly positive values only."""
    v = _daytime(df, start_hour, end_hour)['ws10m'].to_numpy()
    return v[v > 0.0]


def monthly_daytime_temperature(df):
    """Mean daytime (06-18 h) air temperature per calendar month, degC."""
    day = _daytime(df, DAY_START_HOUR, DAY_END_HOUR)
    means = day.groupby(day['time'].dt.month)['t2m'].mean()
    return {int(month): float(value) for month, value in means.items()}


def save_selected_distribution(csv_path, distribution, shape, loc, scale):
    """Write the selected wind distribution (scipy.stats name and parameters)."""
    pd.DataFrame({'distribution': [distribution], 'shape': [shape], 'loc': [loc],
                  'scale': [scale]}).to_csv(csv_path, index=False)


def load_selected_distribution(csv_path):
    """Frozen scipy.stats distribution from 02_selected_wind_distribution.csv."""
    row = pd.read_csv(csv_path).iloc[0]
    family = getattr(stats, str(row['distribution']))
    return family(float(row['shape']), loc=float(row['loc']), scale=float(row['scale']))
