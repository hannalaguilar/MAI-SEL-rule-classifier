"""
Implementation of the CN2 algorithm
"""
from dataclasses import dataclass, field
from typing import Union, List
from pathlib import Path
import numpy as np
import pandas as pd

from src.tools import Rule, DataCSV
from src import tools


@dataclass
class Complex:
    complex: List[Rule]
    data: DataCSV
    n_classes: int
    recall: float = field(init=False)
    precision: float = field(init=False)

    def __post_init__(self):
        self.recall, self.precision = tools.get_recall_precision(self.data,
                                                                 self.complex)

    def __str__(self) -> str:
        conditions = [str(rule) for rule in self.complex]
        cpx = ' AND '.join(conditions)
        return f'IF {cpx} THEN {self.consequent}'

    @property
    def class_dist(self) -> np.ndarray:
        return tools.calculate_class_dist(self.complex,
                                          self.data.df[self.covered_examples],
                                          self.n_classes)

    @property
    def consequent(self) -> int:
        return np.argmax(self.class_dist)

    @property
    def covered_examples(self) -> pd.Series:
        return tools.get_covered_examples(self.data.df, self.complex)

    @property
    def output_data(self) -> DataCSV:
        idx1 = self.data.df.index
        idx2 = self.data.df[self.covered_examples].index
        idx_intersection = idx1.difference(idx2, sort=False)
        data = self.data.df.loc[idx_intersection, :]
        return DataCSV(data.reset_index().drop('index', axis=1))

    @property
    def complex_as_list(self):
        return [self.complex, self.precision, self.recall]


@dataclass
class CN2Classifier:
    k_beam: int = 10
    min_covered_examples: int = 1

    def stopping(self, df: pd.DataFrame) -> bool:
        return df.shape[0] < self.min_covered_examples

    def find_rules(self, data: DataCSV) -> List[Complex]:
        selectors = data.selectors
        n_classes = data.n_classes
        rule_list = []
        while not self.stopping(data.df):
            new_cpx = self.find_best_complex(data,
                                             selectors,
                                             n_classes)
            if new_cpx is None:
                break

            new_cpx = Complex(new_cpx, data, n_classes)
            data = new_cpx.output_data
            rule_list.append(new_cpx)

        return rule_list

    def find_best_complex(self,
                          data: DataCSV,
                          selectors: List[Rule],
                          n_classes: int) -> List[Rule]:
        star = [[]]
        best_cpx = None
        while star:
            new_star = []
            for cpx in star:
                for sel in selectors:
                    new_cpx = cpx + [sel]
                    if new_cpx not in star and tools.is_significant(new_cpx,
                                                                    data.df,
                                                                    n_classes):
                        new_star.append(new_cpx)
            new_star = tools.remove_null_rules(new_star)
            complex_quality = tools.complex_score_list(new_star, data,
                                                       n_classes)
            new_star = tools.sort_by_complex(complex_quality, new_star)

            if not new_star:
                break
            while len(new_star) > self.k_beam:
                new_star = new_star[:self.k_beam]
            for cpx in new_star:
                if best_cpx is None or tools.user_criteria(data,
                                                           n_classes,
                                                           cpx,
                                                           best_cpx):
                    best_cpx = cpx
            star = new_star

        return best_cpx


def run_cn2(data_path: Union[str, Path],
            continuous_attributes: List[str],
            verbose: bool = True) -> List:
    # Load dataframe
    df = pd.read_csv(data_path, index_col=0)
    df, le = tools.preprocess_data(df, continuous_attributes)
    data = DataCSV(df)

    # Find the rule_list
    model = CN2Classifier()
    rule_list = model.find_rules(data)

    if verbose:
        for rule in rule_list:
            print(rule)

    rule_list = [r.complex_as_list for r in rule_list]

    return rule_list
