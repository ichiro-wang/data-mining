# Author: Arash Khoeini
# Email: akhoeini@sfu.ca
# Written for SFU CMPT 459

from typing import Optional, Sequence, Mapping
import numpy as np
import pandas as pd
from node import Node


class DecisionTree(object):
    def __init__(
        self,
        criterion: Optional["str"] = "gini",
        max_depth: Optional[int] = None,
        min_samples_split: Optional[int] = None,
    ):
        """
        :param criterion:
            The function to measure the quality of a split. Supported criteria are “gini” for the Gini
            impurity and “entropy” for the information gain.
        :param max_depth:
            The maximum depth of the tree.
        :param min_samples_split:
            The minimum number of samples required to be at a leaf node
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.criterion_func = self.entropy if criterion == "entropy" else self.gini
        self.saved_threshold = None  # added attribute to save best threshold in gini

    def fit(self, X: pd.DataFrame, y: pd.Series) -> float:
        """
        :param X: data
        :param y: label column in X
        :return: accuracy of training dataset
        HINT1: You use self.tree to store the root of the tree
        HINT2: You should use self.split_node to split a node
        """
        # Your code
        node_size = len(y)
        node_class = y.mode().iloc[0]
        depth = 0
        single_class = y.nunique() == 1

        root = Node(node_size, node_class, depth, single_class)

        self.tree = root

        self.split_node(node=root, X=X, y=y)

        return self.evaluate(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        :param X: data
        :return: predict the class for X.
        HINT1: You can use get_child_node method of Node class to traverse
        HINT2: You can use the mode of the class of the leaf node as the prediction
        HINT3: start traversal from self.tree
        """
        predictions = []
        # Your code

        # iterate over each row
        for i, row in X.iterrows():
            node = self.tree
            # traverse until we hit a leaf
            while not node.is_leaf:
                feature_value = row[node.name]
                child = node.get_child_node(feature_value)

                if not child:
                    break

                node = child
            predictions.append(node.node_class)

        return np.array(predictions)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> int:
        """
        :param X: data
        :param y: labels
        :return: accuracy of predictions on X
        """
        preds = self.predict(X)
        acc = sum(preds == y) / len(preds)
        return acc

    def split_node(self, node: Node, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Splits the data in the node into child nodes based on the best feature.

        :param node: the current node to split
        :param X: data in the node
        :param y: labels in the node
        :return: None
        HINT1: Find the best feature to split the data in 'node'.
        HINT2: Use the criterion function (entropy/gini) to score the splits.
        HINT3: Split the data into child nodes
        HINT4: Recursively split the child nodes until the stopping condition is met (e.g., max_depth or single_class).
        """
        # your code
        if self.stopping_condition(node=node):
            return

        if node.size < self.min_samples_split:
            return

        best_criterion_value = float("inf")
        best_feature = None
        best_threshold = None

        features = X.columns.values

        for feature in features:
            criterion_value = (
                self.gini(X, y, feature)
                if self.criterion == "gini"
                else self.entropy(X, y, feature)
            )
            if criterion_value < best_criterion_value:
                best_criterion_value = criterion_value
                best_feature = feature
                is_numerical = pd.api.types.is_numeric_dtype(X[feature])
                if is_numerical:
                    best_threshold = self.saved_threshold

        node.name = best_feature

        is_numerical = pd.api.types.is_numeric_dtype(X[best_feature])
        node.is_numerical = is_numerical

        if is_numerical:
            node.threshold = best_threshold
            children = {}

            X_l = X[X[best_feature] < best_threshold]
            X_ge = X[X[best_feature] >= best_threshold]

            if len(X_l) == 0 or len(X_ge) == 0:
                return

            y_l = y[X[best_feature] < best_threshold]
            y_ge = y[X[best_feature] >= best_threshold]

            child_depth = node.depth + 1

            child_l_size = len(X_l)
            child_l_class = y_l.mode().iloc[0]
            child_l_single_class = y_l.nunique() == 1

            child_l = Node(
                child_l_size, child_l_class, child_depth, child_l_single_class
            )

            child_ge_size = len(X_ge)
            child_ge_class = y_ge.mode().iloc[0]
            child_ge_single_class = y_ge.nunique() == 1

            child_ge = Node(
                child_ge_size, child_ge_class, child_depth, child_ge_single_class
            )

            self.split_node(child_l, X_l, y_l)
            self.split_node(child_ge, X_ge, y_ge)

            children["l"] = child_l
            children["ge"] = child_ge

            node.set_children(children)

        else:
            children = {}
            for v in X[best_feature].unique():
                X_v = X[X[best_feature] == v]
                y_v = y[X[best_feature] == v]

                child_size = len(X_v)
                child_class = y_v.mode().iloc[0]
                child_depth = node.depth + 1
                child_single_class = y_v.nunique() == 1

                child = Node(child_size, child_class, child_depth, child_single_class)

                X_child = X_v.drop(best_feature, axis=1)

                self.split_node(child, X_child, y_v)

                children[v] = child

            node.set_children(children)

    def stopping_condition(self, node: Node) -> bool:
        """
        Checks if the stopping condition for splitting is met.

        :param node: The current node
        :return: True if stopping condition is met, False otherwise
        """
        # Check if the node is pure (all labels are the same)
        # Check if the maximum depth is reached
        if node.single_class:
            return True
        if self.max_depth is not None and node.depth >= self.max_depth:
            return True
        if node.size < self.min_samples_split:
            return True
        return False

    def gini(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns gini index of the give feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get gini score
        :return:
        """
        is_categorical = pd.api.types.is_string_dtype(X[feature])
        if is_categorical:
            # if feature is categorical

            gini_x = 0
            # each unique value forms a partition
            values = X[feature].unique()
            for v in values:
                # subset of X where feature = v
                X_v = X[X[feature] == v]
                # subset of y where feature = v
                y_v = y[X[feature] == v]

                # gini(T) = 1 - Σp_label^2 for each class (<=50k and >50k)
                # p_label is relative freq. of each label
                sum_p_squared = 0
                for label in y_v.unique():
                    p_label = (y_v == label).sum() / len(y_v)
                    sum_p_squared += p_label**2

                # relative frequency of v
                weight = len(X_v) / len(X)

                gini_v = 1 - sum_p_squared
                gini_x += weight * gini_v

            return gini_x
        else:
            # if feature is numerical

            best_gini = float("inf")
            best_threshold = 0

            # simple implementation where we set each unique value as a threshold
            thresholds = X[feature].unique()

            for t in thresholds:
                # split X into subsets where one is < t and one is >= t
                X_l = X[X[feature] < t]
                X_ge = X[X[feature] >= t]

                # if either subset is empty then we get division by 0 later on
                # just skip this value
                if len(X_l) == 0 or len(X_ge) == 0:
                    continue

                # same for y
                y_l = y[X[feature] < t]
                y_ge = y[X[feature] >= t]

                # calculating gini index of left subset
                sum_p_squared = 0
                for label in y_l.unique():
                    p_label = (y_l == label).sum() / len(y_l)
                    sum_p_squared += p_label**2
                gini_l = 1 - sum_p_squared
                weight_l = len(X_l) / len(X)

                # calculating gini index of right subset
                sum_p_squared = 0
                for label in y_ge.unique():
                    p_label = (y_ge == label).sum() / len(y_ge)
                    sum_p_squared += p_label**2
                gini_ge = 1 - sum_p_squared
                weight_ge = len(X_ge) / len(X)

                # gini index of this threshold
                gini_x = weight_l * gini_l + weight_ge * gini_ge

                if gini_x < best_gini:
                    best_gini = gini_x
                    best_threshold = t

            self.saved_threshold = best_threshold

            return best_gini

    def entropy(self, X: pd.DataFrame, y: pd.Series, feature: str) -> float:
        """
        Returns entropy of the given feature

        :param X: data
        :param y: labels
        :param feature: name of the feature you want to use to get entropy score
        :return:
        """

        # get all unique values of this feature to iterate over

        is_categorical = pd.api.types.is_string_dtype(X[feature])
        if is_categorical:
            values = X[feature].unique()
            entropy = 0

            for v in values:
                # subset of X where feature = v
                X_v = X[X[feature] == v]
                # subset of y where feature = v
                y_v = y[X[feature] == v]

                # relative frequency of v
                weight = len(X_v) / len(X)

                # calculate entropy of v
                ent_v = 0
                # y_v is <=50k or >50k
                for label in y_v.unique():
                    # take relative freq. of <=50k or >50k
                    p_label = (y_v == label).sum() / len(y_v)
                    ent_v -= p_label * np.log2(p_label)

                entropy += weight * ent_v

            return entropy
        else:
            best_entropy = float("inf")
            best_threshold = None

            # simple implementation where we set each unique value as a threshold
            thresholds = X[feature].unique()

            for t in thresholds:
                X_l = X[X[feature] < t]
                X_ge = X[X[feature] >= t]

                if len(X_l) == 0 or len(X_ge) == 0:
                    continue

                y_l = y[X[feature] < t]
                y_ge = y[X[feature] >= t]

                ent_l = 0
                for label in y_l.unique():
                    p_label = (y_l == label).sum() / len(y_l)
                    ent_l -= p_label * np.log2(p_label)
                weight_l = len(X_l) / len(X)

                ent_ge = 0
                for label in y_ge.unique():
                    p_label = (y_ge == label).sum() / len(y_ge)
                    ent_ge -= p_label * np.log2(p_label)
                weight_ge = len(X_ge) / len(X)

                entropy = weight_l * ent_l + weight_ge * ent_ge

                if entropy < best_entropy:
                    best_entropy = entropy
                    best_threshold = t

            self.saved_threshold = best_threshold

            return best_entropy
