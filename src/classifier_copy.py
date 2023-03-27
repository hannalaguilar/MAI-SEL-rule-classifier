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
    data_path: Union[Path, str]
    dataframe: pd.DataFrame = field(init=False)
    attributes_names: Union[List[str], List[int]] = field(init=False)
    target_names: str = field(init=False)
    n_target: int = field(init=False)
    n_attributes: int = field(init=False)
    X_df: np.ndarray = field(init=False)
    Y_df: np.ndarray = field(init=False)

    def __post_init__(self):
        self.dataframe = pd.read_csv(self.data_path, index_col=0)
        self.attributes_names = self.dataframe.iloc[:, :-1].columns
        self.target_names = self.dataframe.iloc[:, -1].unique()
        self.n_attributes = len(self.attributes_names)

        self.X_df = self.dataframe.iloc[:, :-1].values
        self.Y_df = self.dataframe.iloc[:, -1].values

        if self.Y_df.dtype == 'float64':
            self.Y_df = self.Y_df.astype('int64')

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
    verbose: bool = True
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

    def find_rules(self, data: DataCSV):
        X = data.X_df
        Y: data.Y_df
        initial_class_dist = calculate_class_dist(Y, data.)
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
                    best_rule = Rule(attribute, value, np.argmax(class_dist))
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
