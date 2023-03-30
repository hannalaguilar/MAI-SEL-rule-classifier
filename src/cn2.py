from dataclasses import dataclass
from typing import Union, List, Optional
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


# @dataclass
# class Complex:
#     rules: List[Rule]
#
#     def __str__(self):
#         conditions = [str(rule) for rule in self.rules]
#         complex = ' AND '.join(conditions)
#         return f'IF {complex}'


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


@dataclass
class CN2Classifier:
    k_beam: int = 10
    min_covered_examples: int = 1

    def stopping(self, X: np.ndarray) -> bool:
        return X.shape[0] < self.min_covered_examples

    def find_rules(self, data: DataCSV):
        rule_list = []
        while not self.stopping(X):
            new_rule = self.find_best_complex(data)
            if new_rule is None:
                break
            X, Y = self._remove_covered_examples(X, Y, new_rule)
            rule_list.append(new_rule)

        return rule_list

    def find_best_complex(self,
                          data: DataCSV) -> Optional[List[Rule]]:
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
            while len(new_star) > self.k_beam:
                new_star.pop(0)
            for cpx in new_star:
                if best_cpx is None:
                    best_cpx = cpx
            star = new_star
        print(count)
        return best_cpx



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
    model = CN2Classifier()

    print(model.find_best_complex(data))
