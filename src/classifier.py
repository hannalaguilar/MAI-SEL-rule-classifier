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
class Rule:
    attribute: int
    value: float
    consequent: Optional[int]
    complex_quality: Optional[float] = None
    is_significance: Optional[bool] = None

    def __str__(self):
        antecedent_str = f'X{self.attribute}={self.value}'
        return f'IF {antecedent_str} THEN y={self.consequent}'

    def satisfies_conditions(self, X: np.ndarray) -> np.ndarray[bool]:
        return X[:, self.attribute] == self.value

    def class_dist(self,
                   Y: np.ndarray,
                   covered_examples: np.ndarray[bool]) -> np.ndarray[int]:
        return calculate_class_dist(Y[covered_examples], np.max(Y) + 1)



@dataclass
class CN2Classifier:
    name: str = 'cn2'
    complex_quality_evaluator: str = 'entropy'
    test_significance: str = 'likelihood ratio statistic'
    verbose: bool = True
    n_bins: int = 3
    max_rules: int = 10
    min_covered_examples: int = 1


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
            new_rule = self.find_best_complex(X, Y, rule_list, initial_class_dist)
            if new_rule is None:
                break
            X, Y = self._remove_covered_examples(X, Y, new_rule)
            rule_list.append(new_rule)
            if len(rule_list) == self.max_rules:
                break
        return rule_list

    def find_best_complex(self,
                    X: np.ndarray,
                    Y: np.ndarray,
                    rule_list: List[Rule],
                    init_class_dist: np.ndarray):

        best_complex = None
        best_complex_quality = 0

        for attribute in range(X.shape[1]):
            for value in np.unique(X[:, attribute]):
                rule = Rule(attribute, value, None)
                covered_examples = rule.satisfies_conditions(X)
                class_dist = rule.class_dist(Y, covered_examples)
                rule.complex_quality = self._entropy_measure(class_dist)
                lrs, p = self._get_significance(class_dist, init_class_dist)
                rule.is_significance = self._is_significance(lrs, p)

                if (rule.complex_quality > best_complex_quality) and rule.is_significance:
                    best_complex = rule

        return best_complex

    @staticmethod
    def _entropy_measure(class_dist: np.ndarray) -> float:
        class_dist = class_dist[class_dist != 0]
        class_dist /= class_dist.sum()
        class_dist *= -np.log2(class_dist)
        return -class_dist.sum()

    @staticmethod
    def _get_significance(class_dist: np.ndarray,
                          expected_dist: np.ndarray) -> Tuple[float, float]:

        # if there is a 0 value, replace for a low value
        class_dist[class_dist == 0] = 1e-4
        expected_dist[expected_dist == 0] = 1e-4
        expected_dist = (class_dist.sum() / expected_dist.sum()) * expected_dist

        lrs = 2 * np.sum(class_dist * np.log(class_dist / expected_dist))
        p = 1 - stats.chi2.cdf(lrs, df=1)

        return lrs, p

    @staticmethod
    def _is_significance(lrs: float, p: float, alpha: float = 1) -> bool:
        return lrs > 0 and p <= alpha



    def _evaluate_rule(self, covered_examples, class_dist, num_examples, rule_list):
        num_covered = len(covered_examples)
        if num_covered == 0:
            return 0
        purity = np.max(class_dist) / num_covered
        coverage = num_covered / num_examples
        complexity = len(rule_list) + 1
        return purity * coverage / np.log2(complexity)


#
# from sklearn.datasets import load_iris
#
# data = load_iris()
# X = data.data
# y = data.target
data = DataCSV('../data/titanic.csv')
X = data.X_df
y = data.Y_df
learner = CN2Classifier(min_covered_examples=1, max_rules=10)
rules = learner.find_rules(X, y)

for i, rule in enumerate(rules):
    print(f'Rule {i + 1}: {rule}')


# p = Path('../data/iris.csv')
# model = CN2Classifier(p)
# a = model.processed_data
from Orange.data import Table
from Orange.classification import CN2Learner, CN2UnorderedLearner
data = Table('titanic')
