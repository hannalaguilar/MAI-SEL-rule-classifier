import itertools
from typing import List

import numpy as np
import pandas as pd
from scipy import stats as stats

from src.cn2 import Rule, DataCSV


def remove_null_rules(rules):
    for rule in rules[:]:
        for rule1, rule2 in itertools.combinations(rule, 2):
            if rule1.attribute == rule2.attribute:
                rules.remove(rule)
    return rules


def sort_by_complex(complex_quality: list[tuple[float, int]],
                    new_star: List[Rule]) -> List[Rule]:
    return [x for _, x in sorted(zip(complex_quality,
                                     new_star),
                                 key=lambda pair: pair[0], reverse=True)]


def complex_score(new_star: List[Rule],
                  data: DataCSV) -> list[tuple[float, int]]:
    df = data.dataframe
    n = data.n_classes
    return [(entropy(calculate_class_dist(complex, df, n)),
             len(complex)) for complex in new_star]


def calculate_class_dist(complex: list[Rule],
                         df: pd.DataFrame,
                         n_classes: int) -> np.ndarray:
    Y = df.iloc[:, -1].values
    # covered examples
    covered_examples = np.ones(df.shape[0], dtype=bool)
    for rule in complex:
        covered_examples &= df.loc[:, rule.attribute] == rule.value

    # class distribution applying the complex
    class_dist = np.bincount(Y[covered_examples].astype('int64'),
                             minlength=n_classes).astype('float')

    return class_dist


def is_significant(complex: List[Rule],
                   df: pd.DataFrame,
                   n_classes: int,
                   alpha: float = 0.05) -> bool:
    # init class distribution
    Y = df.iloc[:, -1].values
    init_class_dist = np.bincount(Y.astype('int64'),
                                  minlength=n_classes).astype('float')

    # covered examples
    class_dist = calculate_class_dist(complex, df, n_classes)

    init_class_dist[init_class_dist == 0] = 1e-5
    class_dist[class_dist == 0] = 1e-5

    _init_class_dist = (class_dist.sum() /
                        init_class_dist.sum()) * init_class_dist

    lrs = 2 * np.sum(class_dist * np.log(class_dist /
                                         _init_class_dist))

    p = 1 - stats.chi2.cdf(lrs, df=1)

    return lrs > 0 and p <= alpha


def entropy(class_dist: np.ndarray) -> float:
    class_dist = class_dist[class_dist != 0]
    class_dist /= class_dist.sum()
    class_dist *= -np.log2(class_dist)
    return -class_dist.sum()
