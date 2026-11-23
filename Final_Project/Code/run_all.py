# File: run_all.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Run every study script in dependency order, list all produced result files and write a run log
# Inputs: the study scripts 01-12 in this folder (and everything they read from data/)
# Outputs: everything in results/tables and results/figures; results/run_log.txt and a dated copy in
#          results/logs/ (start/end time, git commit, Python and package versions, CPU count, worker
#          processes, time and exit status of each script)
# Author: Ali Khuzam (481014706)
# Date: 2026-11-18

import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version, PackageNotFoundError

from params import VARIANTS, project_variant

script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(script_dir, '..'))
VARIANT = project_variant()                     # PROJECT_VARIANT=v2 writes results_v2/
results_dir = os.path.join(project_dir, VARIANTS[VARIANT][1])
LOG_FILE = os.path.join(results_dir, 'run_log.txt')
LOGS_DIR = os.path.join(results_dir, 'logs')

# 02 (wind distribution) and 03 (vibration curve) feed 04-06; 01 needs neither; 07-10 read 05/06 results; 12 reads 03/04.
SCRIPTS = ('02_fit_wind_distribution.py', '03_gust_vibration_study.py', '01_validate_baseline.py',
           '04_sensitivity_map.py', '05_monte_carlo_risk.py', '06_optimization.py',
           '07_post_analysis.py', '08_search_methods.py',
           '10_step_response.py', '11_export_series.py', '12_pid_gain_check.py')
PACKAGES = ('numpy', 'scipy', 'pandas', 'matplotlib', 'pvlib', 'pytest', 'requests')
N_WORKERS = min(os.cpu_count(), 8)      # same rule as scripts 05, 06 and 08


def git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=project_dir, capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(['git', 'status', '--porcelain', '--', 'code', VARIANTS[VARIANT][0]], cwd=project_dir,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + (' (uncommitted changes in code/ or data/)' if dirty else '')
    except (OSError, subprocess.CalledProcessError):
        return 'not available (no git repository)'


def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return 'not installed'


def main():
    started = datetime.now()
    header = [f'Run log of code/run_all.py'
              + ('' if VARIANT == 'v1' else f' (variant {VARIANT}: {VARIANTS[VARIANT][0]}/ -> {VARIANTS[VARIANT][1]}/)'),
              f'start          : {started:%Y-%m-%d %H:%M:%S}',
              f'git commit     : {git_commit()}',
              f'python         : {platform.python_version()} ({platform.platform()})',
              *[f'{name:15s}: {package_version(name)}' for name in PACKAGES],
              f'cpu count      : {os.cpu_count()}, worker processes N_WORKERS = {N_WORKERS}']
    timings = []
    status = 'completed'
    t_total = time.perf_counter()
    for name in SCRIPTS:
        print(f'\n========== {name} ==========', flush=True)
        t_start = time.perf_counter()
        code = subprocess.run([sys.executable, os.path.join(script_dir, name)]).returncode
        timings.append((name, time.perf_counter() - t_start, code))
        if code != 0:
            status = f'stopped: {name} exited with code {code}'
            break
    total_min = (time.perf_counter() - t_total) / 60

    print('\n=== Run times ===')
    for name, seconds, code in timings:
        print(f'  {name:32s} {seconds / 60:7.1f} min   exit {code}')
    print(f'  {"total":32s} {total_min:7.1f} min')

    ended = datetime.now()
    lines = header + [f'end            : {ended:%Y-%m-%d %H:%M:%S}', f'status         : {status}', '',
                      'script                           time (min)  exit']
    lines += [f'{name:32s} {seconds / 60:10.1f}  {code}' for name, seconds, code in timings]
    lines += [f'{"total":32s} {total_min:10.1f}']
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    os.makedirs(LOGS_DIR, exist_ok=True)
    dated = os.path.join(LOGS_DIR, f'{started:%Y-%m-%d_%H%M}_run_log.txt')
    shutil.copyfile(LOG_FILE, dated)

    print('\n=== Files in results/ ===')
    for sub in ('tables', 'figures'):
        folder = os.path.join(results_dir, sub)
        for root, _, files in os.walk(folder):
            for file_name in sorted(files):
                print(f'  {os.path.relpath(os.path.join(root, file_name), project_dir)}')
    print(f"  run log: {os.path.relpath(LOG_FILE, project_dir).replace(os.sep, '/')} (copy: {os.path.relpath(dated, project_dir)})")
    if status != 'completed':
        sys.exit(1)


if __name__ == '__main__':
    main()
