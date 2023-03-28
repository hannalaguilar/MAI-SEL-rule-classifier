import copy
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer


def calculate_class_dist(Y: np.ndarray, minlength: int) -> np.ndarray:
    return np.bincount(Y, minlength=minlength)


@dataclass
class DataCSV:
    data_path: Union[Path, str]
    dataframe: pd.DataFrame = field(init=False)
    attributes_names: Union[List[str], List[int]] = field(init=False)
    target_names: str = field(init=False)
    n_attributes: int = field(init=False)
    n_target: int = field(init=False)
    target: int = field(init=False)
    X_df: np.ndarray = field(init=False)
    Y_df: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dataframe = pd.read_csv(self.data_path, index_col=0)
        self.attributes_names = self.dataframe.iloc[:, :-1].columns
        self.target_names = self.dataframe.iloc[:, -1].unique()
        self.n_attributes = len(self.attributes_names)
        self.n_target = len(self.target_names)
        self.X_df = self.dataframe.iloc[:, :-1].values
        self.Y_df = self.dataframe.iloc[:, -1].values

        if self.Y_df.dtype == 'float64':
            self.Y_df = self.Y_df.astype('int64')

    #
    # def _preprocess_data(self) -> None:
    #     # copy dataframe
    #     processed_data = self.data.copy()
    #     X_df = processed_data.iloc[:, :-1]
    #
    #     # get attributes types
    #     continuous_attribute = X_df.select_dtypes(include='number').columns
    #     cat_attribute = X_df.select_dtypes(exclude='number').columns
    #
    #     if self.verbose:
    #         print(f' Num. continuous attributes: {len(continuous_attribute)}')
    #         print(f' Num. categorical attributes: {len(cat_attribute)}')
    #
    #     # check if continuous attributes exists
    #     if len(continuous_attribute) != 0:
    #         for attribute in continuous_attribute:
    #             dis = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal',
    #                                    strategy='quantile')
    #             attribute_values = processed_data[attribute].values.\
    #                 reshape(-1, 1)
    #             discretized_attribute = dis.fit_transform(attribute_values)
    #             processed_data[attribute] = discretized_attribute.ravel()
    #     self.processed_data = processed_data


@dataclass
class Selector:
    attribute: int
    operator: str
    value: float

    OPERATORS = ['==', '!=']

    def __str__(self) -> str:
        return f'Attribute {self.attribute}{self.operator}{self.value}'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Rule:
    selectors: List[Selector]
    X: np.ndarray = None
    Y: np.ndarray = None
    class_dist: np.ndarray = None
    consequent: Optional[int] = None
    complex_quality: Optional[float] = None
    is_significance: Optional[bool] = None

    def covered_example(self):
        covered_examples = np.ones(self.X.shape[0], dtype=bool)
        for selector in self.selectors:
            covered_examples &= X[:, selector.attribute] == selector.value


    # def __post_init__(self):
    #     self.class_dist = self._get_class_dist()
    #     self.complex_quality = self._entropy_measure()


    # def __str__(self):
    #     conditions = [s.selector for s in self.selectors]
    #     condition_str = ' AND '.join(conditions)
    #
    #     if self.consequent is not None:
    #         return f'If {condition_str} THEN y={self.consequent}'
    #
    #
    # def satisfies_conditions(self, X: np.ndarray) -> np.ndarray[bool]:
    #     return X[:, self.attribute] == self.value
    #
    # def _get_class_dist(self,
    #                     Y: np.ndarray,
    #                     covered_examples: np.ndarray[bool]) -> np.ndarray[int]:
    #     for selector in self.selectors:
    #         return calculate_class_dist(Y[covered_examples], np.max(Y) + 1)
    #
    #
    # def _entropy_measure(self) -> float:
    #     class_dist = self.class_dist[self.class_dist != 0]
    #     class_dist /= class_dist.sum()
    #     class_dist *= -np.log2(class_dist)
    #     return -class_dist.sum()
    #
    # @staticmethod
    # def _get_significance(class_dist: np.ndarray,
    #                       expected_dist: np.ndarray) -> Tuple[float, float]:
    #     # if there is a 0 value, replace for a low value
    #     class_dist[class_dist == 0] = 1e-4
    #     expected_dist[expected_dist == 0] = 1e-4
    #     expected_dist = (class_dist.sum() / expected_dist.sum()) * expected_dist
    #
    #     lrs = 2 * np.sum(class_dist * np.log(class_dist / expected_dist))
    #     p = 1 - stats.chi2.cdf(lrs, df=1)
    #
    #     return lrs, p
    #
    # @staticmethod
    # def _is_significance(lrs: float, p: float, alpha: float = 1) -> bool:
    #     return lrs > 0 and p <= alpha


@dataclass
class CN2Classifier:
    name: str = 'cn2'
    complex_quality_evaluator: str = 'entropy'
    test_significance: str = 'likelihood ratio statistic'
    beam: int = 5
    max_rule_length: int = 5
    min_covered_examples: int = 1
    verbose: bool = True
    n_bins: int = 3

    def stopping(self, X: np.ndarray) -> bool:
        return X.shape[0] < self.min_covered_examples

    @staticmethod
    def _remove_covered_examples(X: np.ndarray,
                                 Y: np.ndarray,
                                 rule: Rule) -> Tuple[np.ndarray, np.ndarray]:
        covered_examples = rule.satisfies_conditions(X)
        return X[~covered_examples], Y[~covered_examples]

    def find_rules(self, data: DataCSV):
        X = data.X_df
        Y = data.Y_df

        initial_class_dist = calculate_class_dist(Y, data.n_target)
        rule_list = []
        while not self.stopping(X):
            new_rule = self.find_best_complex(X, Y, rule_list,
                                              initial_class_dist)
            if new_rule is None:
                break
            X, Y = self._remove_covered_examples(X, Y, new_rule)
            rule_list.append(new_rule)

        return rule_list

    def find_best_complex(self,
                          X: np.ndarray,
                          Y: np.ndarray,
                          rule_list: List[Rule],
                          init_class_dist: np.ndarray):

        best_rules = None

        # Step 1: Initialize STAR set as simple rules
        star = [Selector(a, op, v) for a in range(X.shape[1])
                for v in np.unique(X[:, a]) for op in Selector.OPERATORS]

        # while len(star) > 0:
        #     new_rules = []
        #     for candidate_rule in star:
        #         new_rules = func(candidate_rule)
        #         rules.extend(new_rules)
        #         for new_rule in new_rules:
        #
        #             if (new_rule.quality > best_rule.quality and
        #                     new_rule.is_significant() and
        #                     new_rule not in rule_list):
        #                 best_rule = new_rule
        #
        #     rules = sorted(rules, key=lambda x: x.quality, reverse=True)
        #     rules = rules[:self.beam]
        # best_rule.create_model()
        # return best_rule


#
# from sklearn.datasets import load_iris
#
# data = load_iris()
# X = data.data
# y = data.target
data = DataCSV('../data/titanic.csv')
X = data.X_df
y = data.Y_df
learner = CN2Classifier(min_covered_examples=1)
rules = learner.find_rules(data)

for i, rule in enumerate(rules):
    print(f'Rule {i + 1}: {rule}')

# p = Path('../data/iris.csv')
# model = CN2Classifier(p)
# a = model.processed_data
from Orange.data import Table
from Orange.classification import CN2Learner, CN2UnorderedLearner

data = Table('titanic')
