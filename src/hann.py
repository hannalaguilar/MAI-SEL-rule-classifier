import pandas as pd
import math
import numpy as np
import scipy.stats as stats
from collections import namedtuple
import itertools

def removing(l):
    for cpx in l[:]:
        for a, b in itertools.combinations(cpx, 2):
            if a.attribute == b.attribute:
                l.remove(cpx)
    return l


def find_best_complex(data, selectors, k_beam, user_criteria, is_significant,
                      complexity_score):
    count = 0
    star = [[]]
    best_cpx = None
    while star:
        new_star = []
        for cpx in star:
            for sel in selectors:
                if cpx != sel:
                    new_cpx = cpx + [sel]
                    if new_cpx not in star and is_significant(new_cpx, data):
                        new_star.append(new_cpx)
                        count += 1
        new_star = removing(new_star)
        new_star = sorted(new_star, key=complexity_score)
        if not new_star:
            break
        while len(new_star) > k_beam:
            new_star.pop(0)
        for cpx in new_star:
            if best_cpx is None:
                best_cpx = cpx

        star = new_star
    print(count)
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


from dataclasses import dataclass, field
from typing import Union, List
@dataclass
class Rule:
    attribute: Union[int, str]
    value: int

    def __str__(self):
        return f'{self.attribute}={self.value}'


@dataclass
class Complex:
    rules: List[Rule]

    def __str__(self):
        conditions = [str(rule) for rule in self.rules]
        complex = ' AND '.join(conditions)
        return f'IF {complex}'


# Load the Titanic dataset
titanic_df = pd.read_csv('../data/titanic.csv', index_col=0)

# Rule = namedtuple('Rule', ['attribute', 'value'])

# Define the set of selectors
# selectors = []
# for col in ['status', 'age', 'sex']:
#     for val in titanic_df[col].unique():
#         selectors.append([col, val])

# lst = selectors
# selectors = rules = [Rule(attribute=lst[i][0], value=lst[i][1]) for i in range(len(lst))]


unique_values = titanic_df.iloc[:, :-1].\
    apply(lambda col: sorted(col.unique())).reset_index().values
selectors = [Rule(attribute=row[0], value=val) for row in
             unique_values for val in row[1]]


# Find the best complex
best_cpx = find_best_complex(titanic_df, selectors, k_beam=10,
                             user_criteria=user_criteria,
                             is_significant=is_statistically_significant,
                             complexity_score=complexity_score)
print(best_cpx)

cpx = Complex([selectors[0], selectors[1]])
print(cpx)