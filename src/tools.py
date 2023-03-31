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
    """
    Represents a rule in the form of 'attribute=value'
    """
    attribute: Union[int, str]
    value: int

    def __str__(self):
        """

        Returns: a string representation of the object in the format
        '{attribute}={value}'.

        """
        return f'{self.attribute}={self.value}'


@dataclass
class DataCSV:
    """
    Represents a Pandas DataFrame. This is the main class for handle data.
    """
    df: pd.DataFrame

    @property
    def target_name(self) -> str:
        """

        Returns: the name of the target column (i.e., the last column).

        """
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


def split_dataframes(df: pd.DataFrame,
                     test_size: float = 0.3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits df into two separate dataframes, one for training and one for
    testing.

    Args:
        df: A Pandas DataFrame representing the data.
        test_size: Test proportion.

    Returns:
        A tuple containing the train and test dataframes.

    """

    X_train, X_test, y_train, y_test = train_test_split(
        df.iloc[:, :-1], df.iloc[:, -1],
        test_size=test_size,
        random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    return train_df, test_df


def preprocess_data(df,
                    continuous_attributes: List[str],
                    n_bins=3):
    """
    This function preprocesses the given dataframe df by performing
    the following steps:
        - Remove columns with more than 50% NaN values.
        - Create a copy of the original dataframe and separate continuous and
        discrete attributes.
        - Replace NaN values in the continuous attributes with their median
         and in the discrete attributes with their mode.
        - Discretize the continuous attributes using the KBinsDiscretizer
        method with n_bins bins and a quantile strategy.
        - Encode the discrete attributes using the OrdinalEncoder.
        - Return the preprocessed dataframe and the fitted OrdinalEncoder.

    Args:
        df: A Pandas DataFrame representing the data.
        continuous_attributes: a list of names of continuous attributes in
        the dataframe.
        n_bins: number of bins to use in the discretization of
        continuous attributes (default=5).

    Returns:
        A tuple containing the preprocessed dataframe and the
        fitted OrdinalEncoder.

    """
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
                               strategy='kmeans')
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
    """
    Removes any rules from the input list that have the same attribute as
    another rule in the list. If two rules have the same attribute,
    the later rule is removed.

    Args:
        rules: A list of Rule objects to filter.

    Returns:
        A filtered list of Rule objects,
        where no two objects have the same attribute.

    """
    for rule in rules[:]:
        for rule1, rule2 in itertools.combinations(rule, 2):
            if rule1.attribute == rule2.attribute:
                rules.remove(rule)
    return rules


def sort_by_complex(complex_quality: list[tuple[float, int]],
                    new_star: List[Rule]) -> List[Rule]:
    """
    Sorts a list of Rule objects by their complexity score,
    in descending order.

    Args:
        complex_quality: A list of tuples containing each rule's complexity
        score and its rule length in the `new_star` list.
        new_star: A list of Rule objects to sort.

    Returns:
        A sorted list of Rule objects, in descending order of their
        complexity score.

    """

    return [x for _, x in sorted(zip(complex_quality, new_star),
                                 key=lambda pair: pair[0], reverse=True)]


def complex_score(df: pd.DataFrame,
                  n_classes: int,
                  complex: List[Rule]) -> float:
    """
    Calculates the complexity score of a set of Rule objects.
    Args:
        df: A Pandas DataFrame representing the data.
        n_classes: The number of unique classes in the target column of `df`.
        complex: A list of Rule objects to calculate the complexity score for.

    Returns:
        The complexity score of the input set of rules,
        as calculated using the entropy of their class distribution.

    """
    class_dist = calculate_class_dist(complex, df, n_classes)
    return entropy(class_dist)


def user_criteria(data: DataCSV, n_classes: int,
                  cpx: List[Rule],
                  best_cpx: List[Rule]) -> bool:
    """
    Compares the quality of two sets of Rule objects, and returns True
    if the first set is of equal or better quality than the second set.

    Args:
        data: A DataCSV object representing the data.
        n_classes: The number of unique classes in the target column of `df`.
        cpx:  A list of Rule objects.
        best_cpx:  A list of Rule objects.

    Returns:
        True if the quality of the `cpx` set of rules is equal to or better
        than the quality of the `best_cpx` set of rules, False otherwise.

    """
    df = data.df
    cpx_quality = complex_score(df, n_classes, cpx)
    best_cpx_quality = complex_score(df, n_classes, best_cpx)
    return cpx_quality >= best_cpx_quality


def complex_score_list(new_star: List[Rule],
                       data: DataCSV,
                       n_classes) -> list[float, int]:
    """
    Calculates the complex score for each complex object.

    Args:
        new_star: A list of Rule objects to calculate the complex score for.
        data: A DataCSV object representing the data.
        n_classes: The number of unique classes in the target column of `df`.

    Returns:
        A list of tuples, where each tuple contains a float representing the
        complex score for a Rule object.

    """
    df = data.df
    return [complex_score(df, n_classes, complex) for complex in new_star]


def get_covered_examples(df: pd.DataFrame, complex) -> pd.Series:
    """
    Returns a Boolean series indicating which examples in a dataset are
    covered by a set of Rule objects.

    Args:
        df: A Pandas DataFrame representing the dataset.
        complex:  A list of Rule objects representing a set of rules.

    Returns:
        A Boolean series indicating which examples in `df` are covered by the
         `complex` set of rules.

    """
    covered_examples = np.ones(df.shape[0], dtype=bool)
    for rule in complex:
        covered_examples &= df.loc[:, rule.attribute] == rule.value
    return covered_examples


def calculate_class_dist(complex: List[Rule],
                         df: pd.DataFrame,
                         n_classes: int) -> np.ndarray:
    """
    Calculate the class distribution of the covered examples of a complex.

    Args:
        complex: A list of Rule objects representing the complex.
        df: A pandas DataFrame with the data.
        n_classes: The number of unique classes in the target column of `df`.

    Returns:
         A numpy array representing the class distribution of the
         covered examples.

    """
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

    """
    Determines whether a set of Rule objects is statistically
    significant for the given dataset.
    Args:
        complex: A list of Rule objects to evaluate for significance.
        df: A Pandas DataFrame representing the data.
        n_classes: The number of unique classes in the target column of `df`.
        alpha: The significance level for the statistical test.

    Returns:
        True if the input set of rules is statistically significant for
         the dataset, False otherwise.
        The significance is determined using a likelihood ratio test,
        comparing the distribution of class labels covered by the input set
         of rules to the overall distribution of class labels in the dataset.

    """

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
    """

    Args:
        class_dist:

    Returns:

    """
    class_dist = class_dist[class_dist != 0]
    class_dist /= class_dist.sum()
    class_dist *= -np.log2(class_dist)
    return -class_dist.sum()


def get_recall_precision(data: DataCSV, cpx: List[Rule]) -> \
        tuple[float, float]:
    """
    Calculates the recall and precision for a complex 'cpx' with respect to the
    data in 'data'.

    Args:
        data: A DataCSV object representing the data.
        cpx:  The complex for which recall and precision is to be
            calculated.

    Returns:
        A tuple containing the recall and precision of the given complex,
        in that order.

    """
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
