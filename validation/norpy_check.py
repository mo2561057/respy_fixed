"""
Simualte the norpy model with respy package under previously estimated
optimal paramters.
Use this as a sanity check of the model correctness.
"""
import yaml

import respy as rp
import numpy as np
import pandas as pd

from ov_respy_config import TEST_RESOURCES_DIR
from adapter.smm_utils import get_moments

#Get the basic modelstructure
options = yaml.safe_load((TEST_RESOURCES_DIR / f"norpy_estimates.yaml").read_text())
params = pd.read_csv(
        TEST_RESOURCES_DIR / f"norpy_estimates.csv", index_col=["category", "name"]
    )

#Get rid of weird string values
for x in params["value"]:
    x = float(x)

#Simulate the data with  the specified coefficeints
simulate = rp.get_simulate_func(params, options)
df = simulate(params)
moments = pd.DataFrame(dict(get_moments(df)))
