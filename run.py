"""
I experiment with respy in this file
"""
import os

import numpy as np
import pandas as pd
import respy as rp

#Explore the basic model structure
params, options, df = rp.get_example_model("kw_97_base")
df.Lagged_Choice = df.Lagged_Choice.fillna("edu")
df["Ability"] = (
    df.groupby("Identifier").Experience_Edu.transform("first")
    .subtract(7)
    .astype(np.uint8)
)

# Add ability parameters to wage components.
for category in ["wage_a", "wage_b", "wage_mil"]:
    params.loc[(category, "at_least_one_ability"), :] = [
        0.1, np.nan, np.nan, "return to having at least ability level one"
    ]

# Add ability parameters to non-pecuniary components.
for category in ["nonpec_edu", "nonpec_home"]:
    params.loc[(category, "at_least_one_ability"), :] = [
        2000, np.nan, np.nan, "return to having at least ability level one"
    ]

# Add ability parameters to type proobabilities.
for category in ["type_2", "type_3", "type_4"]:
    params.loc[(category, "at_least_one_ability"), :] = [
        0.1, np.nan, np.nan, "return to having at least ability level one"
    ]

# Define the probability for ability levels for the simulation.
for name, val in zip(
    [f"level_{i}" for i in range(5)], [0.00981, 0.0431, 0.201, 0.6702, 0.0759]
):
    params.loc[("ability", name), :] = [
        val, np.nan, np.nan, "Probability of having the specified ability level"
    ]

