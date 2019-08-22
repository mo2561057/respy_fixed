"""
This file contains the first estimation runs of the norpy model with respy.
"""
import os
import yaml

import numpy as np
import pandas as pd
import pybobyqa

#Be careful with paths there is something convoluted !
from submodules.estimagic.estimagic.optimization.optimize import minimize
from adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from ov_respy_config import TEST_RESOURCES_DIR
from estimation.smm_auxiliary import moments_final, weigthing_final
from adapter.smm_utils import get_moments

constraints_estimagic = [{"loc":"shocks",
                          "type":"sdcorr",
                          "case":"uncorrelated"},
                         {"locs":[("nonpec_a","hs_graduate"),("nonpec_home","hs_graduate")],
                          "type":"pairwise_equality"},
                         {"locs":[("nonpec_a","co_graduate"),("nonpec_home","co_graduate")],
                          "type":"pairwise_equality"},
                         {"loc":("wage_a","is_minor"), "type": "fixed", "value":0},
                         {"loc":("nonpec_edu","is_minor"), "type":"fixed", "value":0},
                         {"loc": ("meas_error", "sd_a"), "type": "fixed", "value": 0},
                         {"loc": ("nonpec_home","is_young_adult"),"type": "fixed", "value": 0},
                         {"loc":("type_2","at_least_ten_years_edu"),"type": "fixed", "value": 0 },
                         {"loc": ("type_3", "at_least_ten_years_edu"), "type": "fixed", "value": 0},
                         {"loc": ("type_4", "at_least_ten_years_edu"), "type": "fixed", "value": 0},

                         ]

#Import start values and model spec
options = yaml.safe_load((TEST_RESOURCES_DIR / f"norpy_estimates.yaml").read_text())
params = pd.read_csv(
        TEST_RESOURCES_DIR / f"norpy_estimates.csv", index_col=["category", "name"]
    )

args = (params,
        options,
        moments_final,
        weigthing_final,
        get_moments
        )

adapter_smm = SimulationBasedEstimationCls(*args)

#Specify variables for the optimization
#rslt = adapter_smm.evaluate(adapter_smm.free_params)
rslt = minimize(criterion = adapter_smm.evaluate,
                          params = adapter_smm.free_params,
                          algorithm = "nlopt_bobyqa",
                          constraints = constraints_estimagic)
