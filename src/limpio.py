import copy
import itertools
from pathlib import Path
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass, field
from scipy import stats
import numpy as np
import pandas as pd
import operator
from sklearn.preprocessing import OrdinalEncoder


def calculate_class_dist(Y: np.ndarray, minlength: int) -> np.ndarray:
    return np.bincount(Y, minlength=minlength)


# @dataclass
# class DataCSV:
#     data_path: Union[Path, str]
#     dataframe: pd.DataFrame = field(init=False)
#     attributes_names: Union[List[str], List[int]] = field(init=False)
#     target_name: str = field(init=False)
#     target_dist: np.ndarray = field(init=False)
#     n_attributes: int = field(init=False)
#     n_target: int = field(init=False)
#     raw_target: pd.Series = field(init=False)
#     raw_attributes: pd.DataFrame = field(init=False)
#     X_df: np.ndarray = field(init=False)
#     Y_df: np.ndarray = field(init=False)
#
#     def __post_init__(self):
#         self.dataframe = pd.read_csv(self.data_path, index_col=0)
#         self.attributes_names = self.dataframe.iloc[:, :-1].columns
#         self.attributes_int = list(range(len(self.attributes_names)))
#         self.target_name = self.dataframe.iloc[:, -1].name
#         self.target_dist = self.dataframe.iloc[:, -1].unique()
#         self.n_attributes = len(self.attributes_names)
#         self.n_target = len(self.target_dist)
#         self.raw_target = self.dataframe.iloc[:, -1]
#         self.raw_attributes = self.dataframe.iloc[:, :-1]
#
#         le = OrdinalEncoder()
#         self.Y_df = le.fit_transform(
#             self.raw_target.values.reshape(-1, 1)).ravel()
#         if self.Y_df.dtype == 'float':
#             self.Y_df = self.Y_df.astype('int')
#
#         le = OrdinalEncoder()
#         self.X_df = le.fit_transform(self.raw_attributes)


@dataclass
class Selector:
    attribute: int
    operator: str
    value: float

    OPERATORS = ['==']

    def __str__(self) -> str:
        return f'Attribute {self.attribute}{self.operator}{self.value}'

    def __repr__(self) -> str:
        return str(self)


@dataclass
class Rule:
    selectors: List[Selector]
    X: np.ndarray
    Y: np.ndarray
    init_class_dist: np.ndarray
    attribute_names: List[str]
    covered_examples: np.ndarray = field(init=False)
    class_dist: Optional[np.ndarray] = field(init=False)
    consequent: Optional[np.ndarray] = field(init=False)
    complex_quality: Optional[float] = field(init=False)
    is_significant: Optional[bool] = field(init=False)

    def __post_init__(self):
        self.covered_examples = self._get_covered_example()
        self.class_dist = self._get_class_dist()
        self.complex_quality = self._measure_entropy()
        self.is_significant = self._evaluate_significance()
        self.consequent = self._set_consequence()

    def __str__(self):
        conditions = [
            f'{self.attribute_names[s.attribute]}{s.operator}{s.value}'
            for s in self.selectors]
        condition_str = ' AND '.join(conditions)
        out = f'{self.consequent} {self.class_dist}'
        if self.consequent is not None:
            return f'If {condition_str} THEN y={out}'

    def _get_covered_example(self):
        covered_examples = np.ones(self.X.shape[0], dtype=bool)
        for selector in self.selectors:
            if selector.operator == '==':
                covered_examples &= self.X[:,
                                    selector.attribute] == selector.value
            # elif selector.operator == '!=':
            #     covered_examples = operator.ne(self.X[:, selector.attribute],
            #                                    selector.value)
        return covered_examples

    def _get_class_dist(self) -> np.ndarray[int]:
        return calculate_class_dist(self.Y[self.covered_examples],
                                    np.max(self.Y) + 1).astype('float')

    def _measure_entropy(self) -> float:
        class_dist = self.class_dist[self.class_dist != 0]
        class_dist /= class_dist.sum()
        class_dist *= -np.log2(class_dist)
        if class_dist.sum() == 0:
            return - np.inf
        return -class_dist.sum()

    def _evaluate_significance(self,
                               alpha: float = 0.1) -> bool:
        # if there is a 0 value, replace for a low value
        self.class_dist[self.class_dist == 0] = 1e-4
        self.init_class_dist[self.init_class_dist == 0] = 1e-4
        _init_class_dist = (self.class_dist.sum() /
                            self.init_class_dist.sum()) * self.init_class_dist

        lrs = 2 * np.sum(self.class_dist * np.log(self.class_dist /
                                                  _init_class_dist))
        p = 1 - stats.chi2.cdf(lrs, df=1)

        return lrs > 0 and p <= alpha

    def _set_consequence(self):
        return np.argmax(self.class_dist)


@dataclass
class CN2Classifier:
    k_beam: int = 5
    max_rule_length: int = 5
    min_covered_examples: int = 1
    verbose: bool = True

    @property
    def name(self):
        return 'cn2'

    @property
    def complex_quality_evaluator(self):
        return 'entropy'

    @property
    def test_significance(self):
        return 'likelihood ratio statistic'

    def stopping(self, X: np.ndarray) -> bool:
        return X.shape[0] < self.min_covered_examples

    @staticmethod
    def _remove_covered_examples(X: np.ndarray,
                                 Y: np.ndarray,
                                 rule: Optional[Rule]) -> Tuple[
        np.ndarray, np.ndarray]:
        covered_examples = rule.covered_examples
        return X[~covered_examples], Y[~covered_examples]

    def find_rules(self, data: DataCSV):
        X = data.X_df
        Y = data.Y_df

        initial_class_dist = calculate_class_dist(Y, data.n_target)
        rule_list = []
        while not self.stopping(X):
            new_rule = self.find_best_complex(X,
                                              Y,
                                              data,
                                              rule_list,
                                              initial_class_dist)
            if new_rule is None:
                break
            X, Y = self._remove_covered_examples(X, Y, new_rule)
            rule_list.append(new_rule)

        return rule_list

    def find_best_complex(self,
                          X: np.ndarray,
                          Y: np.ndarray,
                          data: DataCSV,
                          rule_list: Union[List[Rule], Optional[List]],
                          init_class_dist: np.ndarray):

        # Step 1: Initialize STAR set as simple rules
        init_selectors = [Selector(a, op, v) for a in range(X.shape[1])
                          for v in np.unique(X[:, a]) for op in
                          Selector.OPERATORS]
        star_rules = [Rule([selectors], X, Y,
                           init_class_dist, data.attributes_names)
                      for selectors in init_selectors]

        star_rules = sorted(star_rules, key=lambda x: x.complex_quality,
                            reverse=True)
        best_rule = star_rules[0]
        while len(star_rules) > 0:
            candidates = star_rules
            star_rules = []
            for candidate_rule in candidates:
                new_rules = self.refine_rules(X, Y, data,
                                              init_class_dist,
                                              candidate_rule)
                star_rules.extend(new_rules)
                for new_rule in new_rules:
                    if (new_rule.complex_quality > best_rule.complex_quality
                            and new_rule.is_significant and
                            new_rule not in rule_list):
                        best_rule = new_rule
            star_rules = sorted(star_rules, key=lambda x: x.complex_quality, reverse=True)
            star_rules = star_rules[:self.k_beam]
        print(best_rule)
        return best_rule if best_rule not in rule_list else None

    def refine_rules(self, X, Y, data, init_class_dist, candidate_rule):
        used_attributes = []
        for selector in candidate_rule.selectors:
            used_attributes.append(selector.attribute)
        all_attributes = data.attributes_int
        remaining_attributes = list(set(all_attributes) ^
                                    set(used_attributes))

        new_selectors = [Selector(a, op, v) for a in remaining_attributes
                         for v in np.unique(X[:, a]) for op in
                         Selector.OPERATORS]
        new_selectors = list(itertools.product(candidate_rule.selectors,
                                               new_selectors))

        new_rules = [Rule(list(selectors), X, Y,
                          init_class_dist, data.attributes_names)
                     for selectors in new_selectors]
        return new_rules


#data = DataCSV(Path('../data/titanic.csv'))
data = DataCSV('mineral.csv')

algorithm = CN2Classifier()
algorithm.find_rules(data)

# le = OrdinalEncoder()
# data.Y_df = le.fit_transform(data.Y_df.reshape(-1, 1))

init_class_dist = calculate_class_dist(data.Y_df, np.max(data.Y_df) + 1)

# rule = Rule(
#     selectors=[
#         Selector(attribute=0, operator='==', value=1),
#         Selector(attribute=2, operator='==', value=0),
#     ], X=data.X_df, Y=data.Y_df, init_class_dist=init_class_dist,
#     attribute_names=['status', 'age', 'sex'])
#
# print(rule)


# ACCURACY # TODO
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




def get_consequent(self, data, cpx):
    *_, consequent = self.get_recall_precision(data, cpx)
    return consequent