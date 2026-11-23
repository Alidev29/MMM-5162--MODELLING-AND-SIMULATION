import numpy as np
import pandas as pd
import pytest
from scipy import stats
from weather_data import (load_power_hourly, daytime_wind, monthly_daytime_temperature,
                          save_selected_distribution, load_selected_distribution)

POWER_CSV = """-BEGIN HEADER-
NASA/POWER Source Native Resolution Hourly Data
Dates (month/day/year): 01/01/2021 through 12/31/2021 in LST
Location: Latitude  24.4672   Longitude 39.6111
Parameter(s):
WS10M     MERRA-2 Wind Speed at 10 Meters (m/s)
T2M       MERRA-2 Temperature at 2 Meters (C)
-END HEADER-
YEAR,MO,DY,HR,WS10M,T2M
2021,1,1,5,2.10,14.2
2021,1,1,6,2.50,15.0
2021,1,1,12,4.20,24.1
2021,1,1,13,-999,-999
2021,1,1,18,3.00,22.0
2021,1,1,19,1.80,20.0
2021,2,1,12,0.00,26.0
2021,2,1,13,5.10,27.5
"""


@pytest.fixture
def power_csv(tmp_path):
    path = tmp_path / 'power.csv'
    path.write_text(POWER_CSV)
    return str(path)


def test_load_skips_header_and_parses_time(power_csv):
    df = load_power_hourly(power_csv)
    assert list(df.columns) == ['time', 'ws10m', 't2m']
    assert df['time'].iloc[0] == pd.Timestamp('2021-01-01 05:00')
    assert df['ws10m'].iloc[0] == pytest.approx(2.10)


def test_load_drops_fill_values(power_csv):
    df = load_power_hourly(power_csv)
    assert len(df) == 7
    assert (df[['ws10m', 't2m']] > -900).all().all()


def test_daytime_filter(power_csv):
    v = daytime_wind(load_power_hourly(power_csv))
    # hours 6..18 inclusive, strictly positive: 2.5, 4.2, 3.0, 5.1 (0.0 at 12:00 in Feb dropped)
    assert np.allclose(np.sort(v), [2.5, 3.0, 4.2, 5.1])


def test_monthly_temperature_keys(power_csv):
    temps = monthly_daytime_temperature(load_power_hourly(power_csv))
    assert set(temps) <= set(range(1, 13))
    assert temps[1] == pytest.approx(np.mean([15.0, 24.1, 22.0]))


def test_selected_distribution_roundtrip(tmp_path):
    dist = stats.weibull_min(2.1, loc=0.0, scale=4.3)
    path = str(tmp_path / 'sel.csv')
    save_selected_distribution(path, 'weibull_min', 2.1, 0.0, 4.3)
    loaded = load_selected_distribution(path)
    assert loaded.mean() == pytest.approx(dist.mean())
