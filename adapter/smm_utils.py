"""
Include a function that extracts releavnt information from the resulting
df of the simulation.
"""
import numpy as np

from collections import OrderedDict
from estimagic.optimization.utilities import sdcorr_params_to_matrix

def get_moments(analysis_df, is_store=False):
    """This function computes the moments based on a dataframe.
       Maybe we want another function to put the results of the simulation into
       a pd DataFrame altough would suggest that this is faster !
    """
    #Brauchen wir des jetzt überhaupt noch ?
    moments = OrderedDict()
    for group in ["Wage Distribution", "Choice Probability"]:
        moments[group] = OrderedDict()

    # We now add descriptive statistics of the wage distribution. Note that there might be
    # periods where there is no information available. In this case, it is simply not added to
    # the dictionary.

    info = analysis_df.copy().groupby("Period")["Wage"].describe().to_dict()
    #print(analysis_df.copy()[analysis_df["period"]==1]["wages"])
    for period in sorted(analysis_df["Period"].unique().tolist()):
        #if pd.isnull(info["std"][period]):
        #    continue
        moments["Wage Distribution"][period] = []
        for label in ["mean", "std"]:
            moments["Wage Distribution"][period].append(info[label][period])

    # We first compute the information about choice probabilities. We need to address the case
    # that a particular choice is not taken at all in a period and then these are not included in
    # the dictionary. This cannot be addressed by using categorical variables as the categories
    # without a value are not included after the groupby operation.
    info = (
        analysis_df.groupby("Period")["Choice"].value_counts(normalize=True).to_dict()
    )
    for period in sorted(analysis_df["Period"].unique().tolist()):
        moments["Choice Probability"][period] = []
        for choice in ["a","edu","home"]:
            try:
                stat = info[(period, choice)]
            except KeyError:
                stat = 0.00
            moments["Choice Probability"][period].append(stat)

#    info = analysis_df['edu'].groupby('agent').max().value_counts(normalize=True).to_dict()
#    for edu_max in range(30):
#        try:
#            stat = info[edu_max]
#        except KeyError:
#            stat = 0
#       moments['Final Schooling'][edu_max] = [stat]

    return moments

def is_valid_covariance_matrix(sd_corr):
    out = sdcorr_params_to_matrix(sd_corr)
    try:
        np.linalg.cholesky(out)
        return True
    except np.linalg.linalg.LinAlgError:
        return False
