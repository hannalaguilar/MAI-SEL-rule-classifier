import copy
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer


def calculate_class_dist(Y: np.ndarray, minlength: int) -> np.ndarray:
    return np.bincount(Y, minlength=minlength)


@dataclass
class DataCSV:
    data_path: Path
    name_attributes: Union[List[str], List[int]]
    name_target: str
    n_target: int
    n_attributes: int
    X: np.ndarray
    Y: np.ndarray

    # def __post_init__(self):
    #     pipeline = [('Read data: check', self._read_data),
    #                 ('Process data: check', self._preprocess_data)]
    #
    #     for step_name, step_func in pipeline:
    #         if self.verbose:
    #             print(step_name)
    #         step_func()

    # def _read_data(self) -> None:
    #     self.data = pd.read_csv(self.data_path, index_col=0)
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
    # data_path: Path
    # X: np.ndarray
    # Y: np.ndarray
    # target: np.ndarray
    # data: pd.DataFrame = field(init=False)
    # processed_data: pd.DataFrame = field(init=False)
    verbose: str = True
    n_bins: int = 3
    name: str = 'cn2'
    max_rules: int = 10
    min_covered_examples: int = 1

    def stopping(self, X: np.ndarray) -> bool:
        return X.shape[0] < self.min_covered_examples

    def _remove_covered_examples(self,
                                 X: np.ndarray,
                                 Y: np.ndarray,
                                 rule: Rule) -> Tuple[np.ndarray, np.ndarray]:
        covered_examples = rule.satisfies_conditions(X)
        return X[~covered_examples], Y[~covered_examples]

    def find_rules(self, X: np.ndarray, Y: np.ndarray):
        rule_list = []
        while not self.stopping(X):
            new_rule = self.rule_finder(X, Y, rule_list)
            if new_rule is None:
                break
            X, Y = self._remove_covered_examples(X, Y, new_rule)
            rule_list.append(new_rule)
            if len(rule_list) == self.max_rules:
                break
        return rule_list

    def rule_finder(self, X: np.ndarray, Y: np.ndarray, rule_list: List[Rule]):
        best_rule = None
        best_score = -1
        for attribute in range(X.shape[1]):
            for value in np.unique(X[:, attribute]):
                rule = Rule(attribute, value, None)
                covered_examples = rule.satisfies_conditions(X)
                class_dist = rule.class_dist(Y, covered_examples)
                score = self._evaluate_rule(covered_examples, class_dist, Y.shape[0], rule_list)
                if score > best_score:
                    best_rule = Rule(attribute, value, np.argmax(class_dist)[0])
                    best_score = score
        return best_rule

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
#
# learner = CN2Classifier(min_covered_examples=5, max_rules=3)
# rules = learner.find_rules(X, y)
#
# for i, rule in enumerate(rules):
#     print(f'Rule {i + 1}: {rule}')


# p = Path('../data/iris.csv')
# model = CN2Classifier(p)
# a = model.processed_data
from Orange.data import Table
from Orange.classification import CN2Learner, CN2UnorderedLearner
data = Table('titanic')
