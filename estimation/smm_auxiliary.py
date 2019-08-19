"""
Prep the original data.
The weighting matrix is cut out to remove elements that we are less interested in.
Its not clear whether it is matcing exactly but the scale seems to be fine.
"""
import os
import pickle

import numpy as np
import pandas as pd

from ov_respy_config import TEST_RESOURCES_DIR
from collections import OrderedDict

#import moments

moments_import = pickle.load(open(os.path.join(TEST_RESOURCES_DIR,"moments.respy.pkl"),"rb"))

weigthing_import = pickle.load(open(os.path.join(TEST_RESOURCES_DIR,"weighing.respy.pkl"),"rb"))

#transform both objects
#moments
moments_final = OrderedDict()

moments_final["Choice Probability"] = OrderedDict(
    {x:[moments_import["Choice Probability"][x][0]] +
       moments_import["Choice Probability"][x][2:4]
     for x in moments_import["Choice Probability"].keys()})

#moments_final["Wage Distribution"] = OrderedDict(
#    {x:moments_import["Wage Distribution"][x][0] for x in moments_import["Wage Distribution"].keys() }
#)
moments_final["Wage Distribution"] = moments_import["Wage Distribution"]

moments_final["Final Schooling"] = moments_import["Final Schooling"]

#weighting

array_kick =np.concatenate((np.arange(76,80), np.arange(81, len(weigthing_import)-30, 4)))

weigthing_intermed = weigthing_import[:-27,:-27]

#Weigthing matrix for leaving the final schooling out
weigthing_final = np.delete(weigthing_intermed,array_kick,0)
weigthing_final = np.delete(weigthing_final,array_kick,1)

#weigthimg matrix with final schooling
weigthing_schooling = np.delete(weigthing_import[:-2,:-2],array_kick,0)
weigthing_schooling = np.delete(weigthing_schooling,array_kick,1)