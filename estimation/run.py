"""
This file contains the first estimation runs of the norpy model with respy.
"""
import os
import yaml

import numpy as np
import pandas as pd
import pybobyqa

from adapter.SimulationBasedEstimation import SimulationBasedEstimationCls
from ov_respy_config import TEST_RESOURCES_DIR
from estimation.smm_auxiliary import moments_final, weigthing_final
from adapter.smm_utils import get_moments

#Specify non common params to optimize
optim_paras_loc = [
    ("delta", "delta"),
    ("wage_a", "constant"),
    ("wage_a", "exp_edu"),
    ("wage_a", "exp_a"),
    ("wage_a","exp_a_square"),
    ("wage_a","hs_graduate"),
    ("wage_a","co_graduate"),
    ("wage_a","period"),
    ("wage_a","any_exp_a"),
    ("wage_a","work_a_lagged"),
    ("nonpec_a","constant"),
    ("nonpec_a","not_exp_a_lagged"),
    ("nonpec_a","not_any_exp_a"),
    ("nonpec_edu","constant"),
    ("nonpec_edu","is_return_not_high_school"),
    ("nonpec_edu","is_return_high_school"),
    ("nonpec_edu","period"),
    ("nonpec_edu","hs_graduate"),
    ("nonpec_edu","co_graduate"),
    ("nonpec_home","constant"),
    ("nonpec_home","period"),
    ("shocks","sd_a"),
    ("shocks","sd_edu"),
    ("shocks","sd_home"),
    ("shocks","corr_edu_a"),
    ("shocks","corr_home_a"),
    ("shocks","corr_home_edu"),
    ("type_shift","type_2_in_a"),
    ("type_shift","type_2_in_edu"),
    ("type_shift","type_2_in_home"),
    ("type_shift","type_3_in_a"),
    ("type_shift","type_3_in_edu"),
    ("type_shift","type_3_in_home"),
    ("type_shift","type_4_in_a"),
    ("type_shift","type_4_in_edu"),
    ("type_shift","type_4_in_home"),
    ("type_2","up_to_nine_years_edu"),
    ("type_3","up_to_nine_years_edu"),
    ("type_4","up_to_nine_years_edu")
]

fixed_params = [("nonpec_home","is_young_adult"),
                ("wage_a","is_minor"),
                ("nonpec_edu","is_minor")
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
        get_moments,
        optim_paras_loc
        )

adapter_smm = SimulationBasedEstimationCls(*args)

#Specify variables for the optimization
non_common_bounds_lower = np.array([params.loc[x,"lower"] for x in optim_paras_loc])
non_common_bounds_upper = np.array([params.loc[x,"upper"] for x in optim_paras_loc])
common_bounds_lower = np.array([-15000,-15000])
common_bounds_upper = np.array([5000, 50000])

bounds_lower = np.concatenate((common_bounds_lower,non_common_bounds_lower))
bounds_upper = np.concatenate((common_bounds_upper,non_common_bounds_upper))
kwargs = dict()
kwargs["scaling_within_bounds"] = True
kwargs["bounds"] = (bounds_lower, bounds_upper)
kwargs["objfun_has_noise"] = True
# kwargs['maxfun'] = 100
kwargs["maxfun"] = 10e6

#rslt = adapter_smm.evaluate(adapter_smm.free_params)
rslt = pybobyqa.solve(adapter_smm.evaluate, adapter_smm.free_params, **kwargs)