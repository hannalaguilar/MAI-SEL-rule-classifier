def find_best_complex(data: DataCSV,
                      user_criteria,
                      k_beam: int = 10) -> Optional[List[Rule]]:
    selectors = data.selectors
    count = 0
    star = [[]]
    best_cpx = None
    while star:
        new_star = []
        for cpx in star:
            for sel in selectors:
                new_cpx = cpx + [sel]
                if new_cpx not in star and is_significant(new_cpx,
                                                          data.dataframe,
                                                          data.n_classes):
                    new_star.append(new_cpx)
                    count += 1
        new_star = remove_null_rules(new_star)
        complex_quality = complex_score(new_star, data)
        new_star = sort_by_complex(complex_quality, new_star)

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



import pandas as pd
import math
import numpy as np
import scipy.stats as stats


def find_best_complex(data, selectors, max_size, user_criteria, is_significant,
                      complexity_score):
    star = set([frozenset()])
    best_cpx = None
    while star:
        new_star = set()
        for cpx in star:
            for sel in selectors:
                new_cpx = cpx.union(sel)
                if new_cpx not in star and is_significant(new_cpx, data):
                    new_star.add(new_cpx)
        new_star = sorted(new_star, key=complexity_score)
        if not new_star:
            break
        while len(new_star) > max_size:
            new_star.pop(0)
        for cpx in new_star:
            if best_cpx is None or user_criteria(cpx, best_cpx, data):
                best_cpx = cpx
        star = new_star
    return best_cpx


def is_statistically_significant(complex, data, alpha=0.05):
    columns = [x[0] for x in complex]
    values = [x[1] for x in complex]
    crosstab = pd.crosstab(index=data[columns], columns=data['Survived'],
                           values=values, aggfunc='count')
    chi2, pval, _, _ = stats.chi2_contingency(crosstab)
    return pval < alpha


def complexity_score(complex):
    return len(complex)


# Load the Titanic dataset
titanic_df = pd.read_csv(
    'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv')

# Define the set of selectors
selectors = []
for col in ['Pclass', 'Sex', 'Age', 'Siblings/Spouses Aboard',
            'Parents/Children Aboard']:
    for val in titanic_df[col].unique():
        selectors.append((col, val))


# Define the user-defined criteria
def user_criteria(new_cpx, old_cpx, data):
    new_cpx_accuracy = accuracy(new_cpx, data)
    old_cpx_accuracy = accuracy(old_cpx, data)
    return new_cpx_accuracy > old_cpx_accuracy


# Define the accuracy function
def accuracy(complex, data):
    columns = [x[0] for x in complex]
    values = [x[1] for x in complex]
    crosstab = pd.crosstab(index=data[columns], columns=data['Survived'],
                           values=values, aggfunc='count')
    if len(crosstab.index) == 1:
        return 1.0
    else:
        return np.min([crosstab.loc[1, 1] / crosstab.loc[1].sum(),
                       crosstab.loc[0, 0] / crosstab.loc[0].sum()])


# Find the best complex
best_cpx = find_best_complex(titanic_df, selectors, max_size=10,
                             user_criteria=user_criteria,
                             is_significant=is_statistically_significant,
                             complexity_score=complexity_score)
print(best_cpx)


import itertools
from collections import namedtuple

Rule = namedtuple('Rule', ['attribute', 'value'])

new_star = [[Rule(attribute='status', value=1.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='status', value=1.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='status', value=2.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='status', value=3.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='status', value=0.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='age', value=0.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='age', value=1.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='sex', value=1.0), Rule(attribute='sex', value=0.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='status', value=1.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='status', value=2.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='status', value=3.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='status', value=0.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='age', value=0.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='age', value=1.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='sex', value=1.0)],
 [Rule(attribute='sex', value=0.0), Rule(attribute='sex', value=0.0)]]


for cpx in new_star[:]:
    print(cpx)
    for a, b in itertools.combinations(cpx, 2):
        if a.attribute == b.attribute:
            print(a.attribute, b.attribute)
            new_star.remove(cpx)