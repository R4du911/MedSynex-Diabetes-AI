import numpy as np


class DecisionTree:
    def __init__(self, min_samples_for_split=2, max_depth=100, max_number_of_features=None):
        self.root = None
        self.min_samples_for_split = min_samples_for_split
        self.max_depth = max_depth
        self.max_number_of_features = max_number_of_features

    def train(self, data, data_target):
        if self.max_number_of_features is None:
            self.max_number_of_features = data.shape[1]
        else:
            self.max_number_of_features = min(data.shape[1], self.max_number_of_features)

        self.root = self.grow_tree(data, data_target)

    def grow_tree(self, data, data_target, depth=0):
        number_of_samples, number_of_features = data.shape
        number_of_target_labels = len(np.unique(data_target))

        if depth >= self.max_depth or number_of_samples < self.min_samples_for_split or number_of_target_labels == 1:
            return Node("Leaf", value=self.determine_most_common_target_label(data_target))

        feature_indices = np.random.choice(number_of_features, self.max_number_of_features, replace=False)

        best_feature, best_threshold = self.determine_best_split(data, data_target, feature_indices)

        left_indices, right_indices = self.split(data[:, best_feature], best_threshold)
        left = self.grow_tree(data[left_indices, :], data_target[left_indices], depth + 1)
        right = self.grow_tree(data[right_indices, :], data_target[right_indices], depth + 1)

        return Node("Node", best_feature, best_threshold, left, right)

    def determine_best_split(self, data, data_target, feature_indices):
        best_information_gain = -1
        split_index, split_threshold = None, None

        for feature_index in feature_indices:
            data_column = data[:, feature_index]
            thresholds = np.unique(data_column)

            for threshold in thresholds:
                information_gain = self.calculate_information_gain(data_target, data_column, threshold)

                if information_gain > best_information_gain:
                    best_information_gain = information_gain
                    split_index = feature_index
                    split_threshold = threshold

        return split_index, split_threshold

    def calculate_information_gain(self, data_target, data_column, threshold):
        parent_entropy = self.calculate_entropy(data_target)

        left_indices, right_indices = self.split(data_column, threshold)
        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        number_of_samples = len(data_target)
        number_left = len(left_indices)
        number_right = len(right_indices)

        left_entropy = self.calculate_entropy(data_target[left_indices])
        right_entropy = self.calculate_entropy(data_target[right_indices])
        child_entropy = (number_left / number_of_samples) * left_entropy + (
                    number_right / number_of_samples) * right_entropy

        return parent_entropy - child_entropy

    def split(self, data_column, split_threshold):
        left_indices = np.argwhere(data_column <= split_threshold).flatten()
        right_indices = np.argwhere(data_column > split_threshold).flatten()
        return left_indices, right_indices

    def calculate_entropy(self, data_target):
        histogram = {}
        for target_label in data_target:
            if target_label in histogram:
                histogram[target_label] += 1
            else:
                histogram[target_label] = 1

        number_of_samples = len(data_target)
        entropy = 0
        for target_label_count in histogram.values():
            p = target_label_count / number_of_samples

            if p > 0:
                entropy -= p * np.log(p)

        return entropy

    def determine_most_common_target_label(self, data_target):
        histogram = {}
        for target_label in data_target:
            if target_label in histogram:
                histogram[target_label] += 1
            else:
                histogram[target_label] = 1

        most_common_target_label = None
        max_target_label_count = -1
        for target_label, target_label_count in histogram.items():
            if target_label_count > max_target_label_count:
                most_common_target_label = target_label
                max_target_label_count = target_label_count

        return most_common_target_label

    def predict(self, data):
        result = []
        for data_entry in data:
            result.append(self.traverse_tree(data_entry, self.root))

        return np.array(result)

    def traverse_tree(self, data, node):
        if node.is_leaf():
            return node.value

        if data[node.feature] <= node.threshold:
            return self.traverse_tree(data, node.left_node)

        return self.traverse_tree(data, node.right_node)


class Node:
    def __init__(self, node_type, feature=None, threshold=None, left_node=None, right_node=None, value=None):
        self.node_type = node_type
        self.feature = feature
        self.threshold = threshold
        self.left_node = left_node
        self.right_node = right_node
        self.value = value

    def is_leaf(self):
        return self.node_type == "Leaf"
