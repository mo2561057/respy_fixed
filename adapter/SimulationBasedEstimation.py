"""
Quick and dirty adapter for smm estimation
"""
import os

import numpy as np

import respy as rp
from ov_respy_config import HUGE_INT
from adapter.smm_utils import is_valid_covariance_matrix


class SimulationBasedEstimationCls:
    """This class manages the distribution of the use requests throughout the toolbox."""

    def __init__(
        self,
        params,
        options,
        moments_obs,
        weighing_matrix,
        get_moments,
        max_evals = HUGE_INT,
    ):


        # Creating a random data array also for the SMM routine allows to align a lot of the
        # designs across the two different estimation strategies.
        self.params = params
        self.options = options
        self.data_array = np.random.rand(8, 8)
        self.weighing_matrix = weighing_matrix
        self.get_moments = get_moments
        self.moments_obs = moments_obs
        self.max_evals = max_evals
        self.simulate_sample = None
        self.num_evals = 1
        self.relevant_coeffs_to_array()


    def evaluate(self, free_params):
        """This method evaluates the criterion function for a candidate parametrization proposed
        by the optimizer.
        we need to translate between the opt dataframe and the model dataframe"""
        print("Hallo")
        self.update_model_spec(self.params, free_params)
        simulate = rp.get_simulate_func(self.params, self.options)
        array_sim = simulate(self.params)

        self.moments_sim = self.get_moments(array_sim)
        stats_obs, stats_sim = [], []

        for group in self.moments_sim.keys():
            for period in range(int(max(self.moments_sim[group].keys()) + 1)):
                if period not in self.moments_sim[group].keys():
                    continue
                if period not in self.moments_obs[group].keys():
                    continue
                stats_obs.extend(self.moments_obs[group][period])
                stats_sim.extend(self.moments_sim[group][period])


        is_valid = (
            len(stats_obs) == len(stats_sim) == len(np.diag(self.weighing_matrix))
        )


        if is_valid:
            print(stats_obs)
            print(stats_sim)
            stats_diff = np.array(stats_obs) - np.array(stats_sim)
            print(stats_diff)
            fval_intermed = np.dot(stats_diff, self.weighing_matrix)
            print(fval_intermed)
            fval = float(np.dot(fval_intermed, stats_diff))
            print(fval)
        else:
            fval = HUGE_INT



        self._logging_smm(stats_obs, stats_sim)
        self.num_evals = self.num_evals + 1
        return fval

    def _logging_smm(self, stats_obs, stats_sim):
        """This method contains logging capabilities that are just relevant for the SMM routine."""
        fname = "monitoring.estimagic.smm.info"

        if self.num_evals == 1 and os.path.exists(fname):
            os.unlink(fname)

        with open(fname, "a+") as outfile:
            fmt_ = "\n\n{:>8}{:>15}\n\n"
            outfile.write(fmt_.format("EVALUATION", self.num_evals))
            fmt_ = "{:>8}" + "{:>15}" * 4 + "\n\n"
            info = ["Moment", "Observed", "Simulated", "Difference", "Weight"]
            outfile.write(fmt_.format(*info))
            for x in enumerate(stats_obs):
                stat_obs, stat_sim = stats_obs[x[0]], stats_sim[x[0]]
                info = [
                    x[0],
                    stat_obs,
                    stat_sim,
                    abs(stat_obs - stat_sim),
                    self.weighing_matrix[x[0], x[0]],
                ]

                fmt_ = "{:>8}" + "{:15.5f}" * 4 + "\n"
                outfile.write(fmt_.format(*info))

    def update_model_spec(self, params, free_params):
        """
        This function updates the model object of the class instance.
        ARGS:
            free_params: np.array of all free paramters
        """
        out_params = params.copy()
        for x in list(out_params.index):
                out_params.loc[x, "value"] = free_params.loc[x, "value"]
        self.params = out_params

    def relevant_coeffs_to_array(self):
        out = self.params.copy()
        self.free_params = out[["value","upper","lower"]]