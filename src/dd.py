import numpy as np


class CN2:
    def __init__(self, min_coverage=10, min_significance=0.5):
        self.min_coverage = min_coverage
        self.min_significance = min_significance
        self.rules = []

    def _get_majority_class(self, y):
        return np.bincount(y).argmax()

    def _get_best_condition(self, X, y, attributes):
        best_attribute = None
        best_value = None
        best_significance = 0
        for attribute in attributes:
            values = np.unique(X[:, attribute])
            for value in values:
                conditions = [(attribute, value)]
                coverage = np.sum(np.apply_along_axis(self._satisfies_conditions, 1, X, conditions))
                if coverage >= self.min_coverage:
                    examples, labels = self._remove_covered_examples(X, y, conditions)
                    majority_class = self._get_majority_class(labels)
                    class_counts = np.bincount(labels)
                    significance = (class_counts[majority_class] / coverage) * np.log2(class_counts.sum() / class_counts[majority_class])
                    if significance > best_significance:
                        best_attribute = attribute
                        best_value = value
                        best_significance = significance
        return (best_attribute, best_value, best_significance)

    def _satisfies_conditions(self, x, conditions):
        for attribute_index, value in conditions:
            if x[attribute_index] != value:
                return False
        return True

    def _remove_covered_examples(self, X, y, conditions):
        mask = np.ones(X.shape[0], dtype=bool)
        for condition in conditions:
            attribute_index, value = condition
            mask &= (X[:, attribute_index] == value)
        return X[~mask], y[~mask]

    def fit(self, X, y, attributes):
        while True:
            best_attribute, best_value, best_significance = self._get_best_condition(X, y, attributes)
            if best_attribute is None:
                break
            rule = {
                "conditions": [(best_attribute, best_value)],
                "label": self._get_majority_class(y)
            }
            self.rules.append(rule)
            X, y = self._remove_covered_examples(X, y, rule["conditions"])

    def predict(self, X):
        y_pred = np.empty(X.shape[0])
        for i, x in enumerate(X):
            for rule in self.rules:
                if self._satisfies_conditions(x, rule["conditions"]):
                    y_pred[i] = rule["label"]
                    break
            else:
                y_pred[i] = self._get_majority_class(y)
        return y_pred


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)

# Create a CN2 classifier and fit it to the training data
clf = CN2(min_coverage=5, min_significance=0.5)
clf.fit(X_train, y_train, attributes=[0, 1, 2, 3])

# Predict the class labels of the test data and calculate the accuracy
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))


# Print the learned rules
for i, rule in enumerate(clf.rules):
    print("Rule {}: If {} is {} then the class is {}".format(i + 1,
                                                             iris.feature_names[rule["conditions"][0][0]],
                                                             rule["conditions"][0][1], iris.target_names[rule["label"]]))