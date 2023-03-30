import pandas as pd
import math
import numpy as np
import scipy.stats as stats
from collections import namedtuple

def find_best_complex(data, selectors, max_size, user_criteria, is_significant,
                      complexity_score):
    star = [[]]
    best_cpx = None
    while star:
        new_star = [[]]
        for cpx in star:
            for sel in selectors:
                if cpx != sel:
                    new_cpx = cpx + list(sel)
                    if new_cpx not in star and is_significant(new_cpx, data):
                        new_star.append(new_cpx)
        new_star = sorted(new_star, key=complexity_score)
        if not new_star:
            break
        while len(new_star) > max_size:
            new_star.pop(0)
        for cpx in new_star:
            if best_cpx is None:
                best_cpx = cpx
        new_star = [l for l in new_star if l]
        star = new_star
    return best_cpx


def is_statistically_significant(complex, data, alpha=0.05):
    # columns = [x[0] for x in complex]
    # values = [x[1] for x in complex]
    # crosstab = pd.crosstab(index=data[columns], columns=data['Survived'],
    #                        values=values, aggfunc='count')
    # chi2, pval, _, _ = stats.chi2_contingency(crosstab)

    pval = 0.001
    return pval < alpha



def complexity_score(complex):
    return len(complex)


# Define the user-defined criteria
def user_criteria(new_cpx, old_cpx, data):
    new_cpx_accuracy = accuracy(new_cpx, data)
    old_cpx_accuracy = accuracy(old_cpx, data)
    return new_cpx_accuracy > old_cpx_accuracy


# Define the accuracy function
def accuracy(complex, data):
    crosstab = pd.crosstab(index=data[list(complex)], columns=data['survive'])
    if len(crosstab.index) == 1:
        return 1.0
    else:
        return np.min([crosstab.loc[1, 1] / crosstab.loc[1].sum(),
                       crosstab.loc[0, 0] / crosstab.loc[0].sum()])


# Load the Titanic dataset
titanic_df = pd.read_csv('../data/titanic.csv', index_col=0)

# Define the set of selectors
selectors = []
for col in ['status', 'age', 'sex']:
    for val in titanic_df[col].unique():
        selectors.append([col, val])


# Find the best complex
best_cpx = find_best_complex(titanic_df, selectors, max_size=10,
                             user_criteria=user_criteria,
                             is_significant=is_statistically_significant,
                             complexity_score=complexity_score)
print(best_cpx)
