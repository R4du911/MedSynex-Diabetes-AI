import numpy as np


class DecisionTree:
    def __init__(self, min_samples_for_split=2, max_depth=100, features_number=None):
        self.root = None
        self.min_samples_for_split = min_samples_for_split
        self.max_depth = max_depth
        self.features_number = features_number


    def train(self, data, data_output):


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
