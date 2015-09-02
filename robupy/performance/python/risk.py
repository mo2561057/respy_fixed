""" This module contains the functions related to the incorporation of
    risk in the model.
"""

# standard library
import numpy as np

# project library
from robupy.performance.python.auxiliary import simulate_emax


''' Public functions
'''


def get_payoffs_risk(num_draws, eps_baseline, period, k, payoffs_ex_ante,
        edu_max, edu_start, mapping_state_idx, states_all, num_periods, emax,
        delta):
    """ Simulate expected future value under risk.
    """
    # Transformation of standard normal deviates to relevant distributions.
    eps_relevant = eps_baseline.copy()
    for j in [0, 1]:
        eps_relevant[:, j] = np.exp(eps_relevant[:, j])

    # Simulate expected future value.
    simulated, payoffs_ex_post, future_payoffs = simulate_emax(num_periods,
        num_draws, period, k, eps_relevant, payoffs_ex_ante, edu_max,
        edu_start, emax, states_all, mapping_state_idx, delta)

    # Finishing
    return simulated, payoffs_ex_post, future_payoffs
