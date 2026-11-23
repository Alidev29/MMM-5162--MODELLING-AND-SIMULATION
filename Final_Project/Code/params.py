# File: params.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Single definition of the random seed, CSV parameter loaders, project paths and the project variant
#          (PROJECT_VARIANT=v2 reads data_v2/ and writes results_v2/).
# Inputs: data/*.csv parameter files (columns Parameter, Value, Unit, Source)
# Outputs: Python dictionaries / DataFrames of parameters; results folders on disk
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17

import os
import pandas as pd

# The only random seed of the project (all randomness uses np.random.default_rng).
SEED = 5162

AXIS_KEYS = ('Jp', 'ks', 'cs')

# Project variants: (parameter folder, results folder). v1 is the submitted model; v2 uses the datasheet of the
# Canadian Solar CS1U-400MS module (data_v2/pv_parameters.csv). The raw weather data in data/raw are shared.
VARIANTS = {'v1': ('data', 'results'), 'v2': ('data_v2', 'results_v2')}


def project_variant():
    """Variant chosen by the environment variable PROJECT_VARIANT: 'v1' (default) or 'v2'."""
    variant = os.environ.get('PROJECT_VARIANT', 'v1')
    if variant not in VARIANTS:
        raise ValueError(f'PROJECT_VARIANT must be one of {sorted(VARIANTS)}, got {variant!r}')
    return variant


def load_parameters(csv_path):
    """Read a Parameter/Value/Unit/Source CSV and return {Parameter: float(Value)}."""
    df_params = pd.read_csv(csv_path)
    return {str(name): float(value) for name, value in zip(df_params['Parameter'], df_params['Value'])}


def load_disturbance_ranges(csv_path):
    """Read disturbance_ranges.csv, indexed by Parameter (columns Nominal, Min, Max, Distribution, SD, Unit, Source)."""
    df_ranges = pd.read_csv(csv_path)
    df_ranges = df_ranges.set_index('Parameter')
    for column in ('Nominal', 'Min', 'Max', 'SD'):
        df_ranges[column] = pd.to_numeric(df_ranges[column], errors='coerce')
    return df_ranges


def project_paths(script_file):
    """Return data/raw/tables/figures paths built from the folder that holds script_file; create them."""
    script_dir = os.path.dirname(os.path.abspath(script_file))
    root_dir = os.path.normpath(os.path.join(script_dir, '..'))
    data_name, results_name = VARIANTS[project_variant()]
    paths = {
        'root': root_dir,
        'data': os.path.join(root_dir, data_name),
        'raw': os.path.join(root_dir, 'data', 'raw'),
        'tables': os.path.join(root_dir, results_name, 'tables'),
        'figures': os.path.join(root_dir, results_name, 'figures'),
    }
    for key in ('data', 'raw', 'tables', 'figures'):
        os.makedirs(paths[key], exist_ok=True)
    return paths


def axis_params(drive, axis):
    """Drive parameters for one axis ('az' or 'el'): shared keys plus Jp, ks, cs without suffix."""
    if axis not in ('az', 'el'):
        raise ValueError(f"axis must be 'az' or 'el', got {axis!r}")
    ax = {key: value for key, value in drive.items()
          if not (key.endswith('_az') or key.endswith('_el'))}
    for key in AXIS_KEYS:
        ax[key] = drive[f'{key}_{axis}']
    return ax
