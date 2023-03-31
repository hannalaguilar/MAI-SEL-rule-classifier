"""
Main tools for performing the CN2 algorithm
"""
import copy
import itertools
from typing import List, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer
from sklearn.model_selection import train_test_split


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
        # unique_values = self.df.iloc[:, :-1]. \
        #     apply(lambda col: sorted(col.unique())).reset_index().values
        unique_values = []
        for col in self.df.iloc[:, :-1].columns:
            v = sorted(self.df[col].unique())
            unique_values.append((col, v))
        return [Rule(attribute=row[0], value=val) for row in
                unique_values for val in row[1]]


def split_dataframes(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame,
pd.DataFrame]:
    val_size = 0.15
    test_size = 0.15
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1],
        test_size=test_size,
        random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid,
                                                          y_train_valid,
                                                          test_size=val_size,
                                                          random_state=42)
    train_df = pd.concat([X_train, y_train], axis=1)
    valid_df = pd.concat([X_valid, y_valid], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, valid_df, test_df


def preprocess_data(df,
                    continuous_attributes: List[str],
                    n_bins=3):
    # remove nan values if there are more than 50%
    n_rows = df.shape[0]
    thresh = int(0.5 * n_rows)
    df = df.dropna(axis='columns', thresh=thresh)

    # copy dataframe
    processed_df = copy.deepcopy(df)
    discrete_attributes = [col for col in df.columns[:-1]
                           if col not in continuous_attributes]

    # replace nan values
    continouous_df = processed_df.loc[:, continuous_attributes]
    continouous_df.fillna(continouous_df.median(), inplace=True)
    discrete_df = processed_df.loc[:, discrete_attributes]
    discrete_df.fillna(discrete_df.mode(), inplace=True)
    processed_df.loc[:, continuous_attributes] = continouous_df.values
    processed_df.loc[:, discrete_attributes] = discrete_df.values

    for attribute in continuous_attributes:
        dis = KBinsDiscretizer(n_bins=n_bins,
                               encode='ordinal',
                               strategy='quantile')
        attribute_values = processed_df[attribute].values. \
            reshape(-1, 1)
        discrete_attribute = dis.fit_transform(attribute_values)
        processed_df[attribute] = discrete_attribute.ravel()
    processed_df = processed_df

    le = OrdinalEncoder()
    processed_df_np = le.fit_transform(processed_df)
    processed_df = pd.DataFrame(data=processed_df_np, columns=df.columns)

    return processed_df, le


def remove_null_rules(rules: List[Rule]) -> List[Rule]:
    for rule in rules[:]:
        for rule1, rule2 in itertools.combinations(rule, 2):
            if rule1.attribute == rule2.attribute:
                rules.remove(rule)
    return rules


def sort_by_complex(complex_quality: list[tuple[float, int]],
                    new_star: List[Rule]) -> List[Rule]:
    return [x for _, x in sorted(zip(complex_quality, new_star),
                                 key=lambda pair: pair[0], reverse=True)]


def complex_score(df: pd.DataFrame,
                  n_classes: int,
                  complex: List[Rule]) -> float:
    class_dist = calculate_class_dist(complex, df, n_classes)
    return entropy(class_dist)


def user_criteria(data: DataCSV, n_classes: int,
                  cpx: List[Rule],
                  best_cpx: List[Rule]) -> bool:
    df = data.df
    cpx_quality = complex_score(df, n_classes, cpx)
    best_cpx_quality = complex_score(df, n_classes, best_cpx)
    return cpx_quality >= best_cpx_quality


def complex_score_list(new_star: List[Rule],
                       data: DataCSV,
                       n_classes) -> list[float, int]:
    df = data.df
    return [complex_score(df, n_classes, complex) for complex in new_star]


def get_covered_examples(df: pd.DataFrame, complex) -> pd.Series:
    covered_examples = np.ones(df.shape[0], dtype=bool)
    for rule in complex:
        covered_examples &= df.loc[:, rule.attribute] == rule.value
    return covered_examples


def calculate_class_dist(complex: List[Rule],
                         df: pd.DataFrame,
                         n_classes: int) -> np.ndarray:
    Y = df.iloc[:, -1].values

    # covered examples
    covered_examples = get_covered_examples(df, complex)

    # class distribution applying the complex
    class_dist = np.bincount(Y[covered_examples].astype('int64'),
                             minlength=n_classes).astype('float')

    return class_dist


def is_significant(complex: List[Rule],
                   df: pd.DataFrame,
                   n_classes: int,
                   alpha: float = 0.1) -> bool:
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


def get_recall_precision(data: DataCSV, cpx: List[Rule]) -> \
        tuple[float, float]:
    # covered examples
    covered_examples = get_covered_examples(data.df, cpx)
    subset = data.df[covered_examples]

    # consequent
    consequent = subset.iloc[:, -1].value_counts().sort_values(
        ascending=False).index[0]

    # original data
    dataset_values = data.df.iloc[:, -1].value_counts()

    # instances satisfying the antecedent of R and the consequent of R
    antecedent_and_consequent = subset[subset.iloc[:, -1] == consequent]
    precision = antecedent_and_consequent.shape[0] / subset.shape[0]
    recall = antecedent_and_consequent.shape[0] / dataset_values[
        consequent]

    return recall, precision
