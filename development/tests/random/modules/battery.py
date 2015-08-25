
""" This modules contains some additional tests that are only used in
long-run development tests.
"""

# standard library
from pandas.util.testing import assert_frame_equal
import pandas as pd
import sys
import os

# project library
from modules.auxiliary import compile_package


# ROBUPY import
sys.path.insert(0, os.environ['ROBUPY'])
from robupy.tests.random_init import generate_init

''' Main
'''
def test_99():
    """ Testing whether the results from a fast and slow execution of the
    code result in identical simulate datasets.
    """
    # Generate random initialization
    generate_init()

    # Initialize containers
    base = None

    for which in ['slow', 'fast']:

        compile_package(which)

        # Simulate the ROBUPY package
        os.system('robupy-solve --simulate --model test.robupy.ini')

        # Load simulated data frame
        data_frame = pd.read_csv('data.robupy.dat')

        # Compare
        if base is None:
            base = data_frame.copy()

        assert_frame_equal(base, data_frame)