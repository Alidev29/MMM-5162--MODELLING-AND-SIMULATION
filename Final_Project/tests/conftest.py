# File: conftest.py
# Course: MMM 5162 Modelling and Simulation
# Purpose: Make the model modules in code/ importable from the tests.
# Author: Ali Khuzam (481014706)
# Date: 2026-11-17
import os
import sys

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(TESTS_DIR, '..', 'code')
DATA_DIR = os.path.join(TESTS_DIR, '..', 'data')
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
