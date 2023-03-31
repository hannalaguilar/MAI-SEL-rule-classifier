from dataclasses import dataclass, field
from typing import Union, List, Optional, NewType
from pathlib import Path
import pandas as pd
import numpy as np

# from src.tools import remove_null_rules, sort_by_complex, complex_score, \
#     is_significant

from src.tools import *


@dataclass
class Rule:
    attribute: Union[int, str]
    value: int

    def __str__(self):
        return f'{self.attribute}={self.value}'


@dataclass
class DataCSV:
    df: pd.DataFrame

    @property
    def target_name(self) -> str:
        return self.df.columns[-1]

    @property
    def n_classes(self) -> int:
        return self.df.iloc[:, -1].nunique()

    @property
    def selectors(self) -> list[Rule]:
        unique_values = self.df.iloc[:, :-1]. \
            apply(lambda col: sorted(col.unique())).reset_index().values
        return [Rule(attribute=row[0], value=val) for row in
                unique_values for val in row[1]]


@dataclass
class Complex:
    complex: List[Rule]
    data: DataCSV
    n_classes: int
    recall: float = field(init=False)
    precision: float = field(init=False)

    def __post_init__(self):
        self.recall, self.precision = get_recall_precision(self.data,
                                                           self.complex)

    def __str__(self):
        conditions = [str(rule) for rule in self.complex]
        cpx = ' AND '.join(conditions)
        return f'IF {cpx} THEN {self.consequent}'

    @property
    def class_dist(self):
        return calculate_class_dist(self.complex,
                                    self.data.df[self.covered_examples],
                                    self.n_classes)

    @property
    def consequent(self):
        return np.argmax(self.class_dist)

    @property
    def covered_examples(self):
        return get_covered_examples(self.data.df, self.complex)

    @property
    def output_data(self):
        idx1 = self.data.df.index
        idx2 = self.data.df[self.covered_examples].index
        ii = idx1.difference(idx2, sort=False)
        data = self.data.df.loc[ii, :]
        return DataCSV(data.reset_index().drop('index', axis=1))


@dataclass
class CN2Classifier:
    k_beam: int = 10
    min_covered_examples: int = 1

    def stopping(self, df: pd.DataFrame) -> bool:
        return df.shape[0] < self.min_covered_examples

    def find_rules(self, data: DataCSV):
        selectors = data.selectors
        n_classes = data.n_classes
        rule_list = []
        while not self.stopping(data.df):
            new_cpx = self.find_best_complex(data,
                                             selectors,
                                             n_classes)
            if new_cpx is None:
                break

            aa = Complex(new_cpx, data, n_classes)

            data = aa.output_data
            rule_list.append(aa)

        return rule_list

    def find_best_complex(self,
                          data: DataCSV,
                          selectors,
                          n_classes) -> List[Rule]:
        star = [[]]
        best_cpx = None
        while star:
            new_star = []
            for cpx in star:
                for sel in selectors:
                    new_cpx = cpx + [sel]
                    if new_cpx not in star and is_significant(new_cpx,
                                                              data.df,
                                                              n_classes):
                        new_star.append(new_cpx)
            new_star = remove_null_rules(new_star)
            complex_quality = complex_score_list(new_star, data, n_classes)
            new_star = sort_by_complex(complex_quality, new_star)

            if not new_star:
                break
            while len(new_star) > self.k_beam:
                new_star = new_star[:self.k_beam]
            for cpx in new_star:
                if best_cpx is None or user_criteria(data, n_classes, cpx,
                                                     best_cpx):
                    best_cpx = cpx
            star = new_star

        return best_cpx


if __name__ == "__main__":
    DATA_PATH = '../data/titanic.csv'
    DF = pd.read_csv(DATA_PATH, index_col=0)
    DATA_INS = DataCSV(DF)

    # Find the rule_list
    model = CN2Classifier()
    rule_list = model.find_rules(DATA_INS)
    for rule in rule_list:
        print(rule)
