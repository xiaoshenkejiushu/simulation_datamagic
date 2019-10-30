# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from simulation_b import *
import numpy as np
import pandas as pd


target_x = []
target_y = []

def gms_function_value(capacity_1,capacity_2,capacity_3,capacity_4):
    """SVC cross validation.
    This function will instantiate a SVC classifier with parameters C and
    gamma. Combined with data and targets this will in turn be used to perform
    cross validation. The result of cross validation is returned.

    Our goal is to find combinations of C and gamma that maximizes the roc_auc
    metric.
    """
    capacity_1 = int(1000000*capacity_1)
    capacity_2 = int(1000000*capacity_2)
    capacity_3 = int(1000000*capacity_3)
    capacity_4 = int(1000000*capacity_4)
    
    actionlist =[capacity_1,capacity_2,capacity_3,capacity_4]
    print('actionlist',actionlist)
    if sum(actionlist) <= 10:
        function_value = callFromGams(actionlist)-10*(10-sum(actionlist))
    else:
        function_value = callFromGams(actionlist)-20*(sum(actionlist)-10)
    print('此时的函数值',function_value)
    #0409新加
    target_y.append(function_value)
    target_x.append(actionlist)
    return function_value

def optimize_gms():
    """Apply Bayesian Optimization to SVC parameters."""
    def gms_value(capacity_1,capacity_2,capacity_3,capacity_4):
        """Wrapper of SVC cross validation.

        Notice how we transform between regular and log scale. While this
        is not technically necessary, it greatly improves the performance
        of the optimizer.
        """
        return gms_function_value(capacity_1,capacity_2,capacity_3,capacity_4)

    optimizer = BayesianOptimization(
        f=gms_value,
        pbounds={"capacity_1": (0.000001,0.000001),"capacity_2": (0.000001,0.000007),"capacity_3": (0.000001,0.000007),"capacity_4": (0.000001,0.000007)},
        verbose=0
    )
    optimizer.maximize(n_iter=50)

    print("Final result:", optimizer.max)



if __name__ == "__main__":
    print(Colours.yellow("--- Optimizing gms ---"))
    optimize_gms()
    print('111')
    df_target = pd.DataFrame()
    df_target['target'] = target_y
    df_target['action'] = target_x
    df_target.to_csv('process_value.csv', index=False)

