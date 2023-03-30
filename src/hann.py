import itertools
from dataclasses import dataclass, field
from typing import Union, List
from pathlib import Path
import pandas as pd
import math
import numpy as np
import scipy.stats as stats


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


@dataclass
class DataCSV:
    data_path: Union[Path, str]

    @property
    def dataframe(self) -> pd.DataFrame:
        return pd.read_csv(self.data_path, index_col=0)

    @property
    def target_name(self):
        return self.dataframe.columns[-1]

    @property
    def n_classes(self):
        return self.dataframe.iloc[:, -1].nunique()

    @property
    def selectors(self) -> List[Rule]:
        unique_values = self.dataframe.iloc[:, :-1]. \
            apply(lambda col: sorted(col.unique())).reset_index().values
        return [Rule(attribute=row[0], value=val) for row in
                unique_values for val in row[1]]


def remove_null_rules(rules):
    for rule in rules[:]:
        for rule1, rule2 in itertools.combinations(rule, 2):
            if rule1.attribute == rule2.attribute:
                rules.remove(rule)
    return rules


def find_best_complex(data: DataCSV,
                      user_criteria,
                      is_significant,
                      complexity_score,
                      k_beam: int = 10):
    selectors = data.selectors
    count = 0
    star = [[]]
    best_cpx = None
    while star:
        new_star = []
        for cpx in star:
            for sel in selectors:
                new_cpx = cpx + [sel]
                if new_cpx not in star and is_significant(new_cpx, data.dataframe, data.n_classes):
                    new_star.append(new_cpx)
                    count += 1
        new_star = remove_null_rules(new_star)
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


def calculate_class_dist(complex, df, n_classes):
    Y = df.iloc[:, -1].values
    # covered examples
    covered_examples = np.ones(df.shape[0], dtype=bool)
    for rule in complex:
        covered_examples &= df.loc[:, rule.attribute] == rule.value

    # class distribution applying the complex
    class_dist = np.bincount(Y[covered_examples].astype('int64'),
                             minlength=n_classes).astype('float')

    return class_dist


def is_statistically_significant(complex: List[Rule],
                                 df: pd.DataFrame,
                                 n_classes: int,
                                 alpha=0.05):

    # init class distribution
    Y = df.iloc[:, -1].values
    init_class_dist = np.bincount(Y.astype('int64'),
                                  minlength=n_classes).astype('float')

    # covered examples
    class_dist = calculate_class_dist(complex, df, n_classes)
    # covered_examples = np.ones(df.shape[0], dtype=bool)
    # for rule in complex:
    #     covered_examples &= df.loc[:, rule.attribute] == rule.value
    #
    # # class distribution applying the complex
    # class_dist = np.bincount(Y[covered_examples].astype('int64'),
    #                          minlength=n_classes).astype('float')

    init_class_dist[init_class_dist == 0] = 1e-5
    class_dist[class_dist == 0] = 1e-5

    _init_class_dist = (class_dist.sum() /
                        init_class_dist.sum()) * init_class_dist

    lrs = 2 * np.sum(class_dist * np.log(class_dist /
                                         _init_class_dist))

    p = 1 - stats.chi2.cdf(lrs, df=1)

    return lrs > 0 and p <= alpha

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


if __name__ == "__main__":
    data_path = '../data/titanic.csv'
    data = DataCSV(data_path)

    # Find the best complex
    BEST_CPX = find_best_complex(data,
                                 user_criteria=user_criteria,
                                 is_significant=is_statistically_significant,
                                 complexity_score=complexity_score)
    print(BEST_CPX)

# # Load the Titanic dataset
# titanic_df = pd.read_csv('../data/titanic.csv', index_col=0)
#
# data = DataCSV('../data/titanic.csv')
# selectors = data.selectors

# Rule = namedtuple('Rule', ['attribute', 'value'])

# Define the set of selectors
# selectors = []
# for col in ['status', 'age', 'sex']:
#     for val in titanic_df[col].unique():
#         selectors.append([col, val])

# lst = selectors
# selectors = rules = [Rule(attribute=lst[i][0], value=lst[i][1]) for i in range(len(lst))]


# unique_values = titanic_df.iloc[:, :-1].\
#     apply(lambda col: sorted(col.unique())).reset_index().values
# selectors = [Rule(attribute=row[0], value=val) for row in
#              unique_values for val in row[1]]
