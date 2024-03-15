import numpy as np
from model.DecisionTree import DecisionTree


class RandomForest:
    def __init__(self, max_tree_number=20, min_samples_for_split=2, max_depth=15, max_number_of_features=None):
        self.trees = []
        self.max_tree_number = max_tree_number
        self.min_samples_for_split = min_samples_for_split
        self.max_depth = max_depth
        self.max_number_of_features = max_number_of_features

    def train(self, data, data_target):
        self.trees = []
        for tree_index in range(self.max_tree_number):
            self.train_tree(data, data_target)

    def train_tree(self, data, data_target):
        tree = DecisionTree(min_samples_for_split=self.min_samples_for_split, max_depth=self.max_depth,
                            max_number_of_features=self.max_number_of_features)
        random_data_samples, random_data_target_samples = self.bootstrap_data(data, data_target)
        tree.train(random_data_samples, random_data_target_samples)
        self.trees.append(tree)

    def bootstrap_data(self, data, data_target):
        number_samples = data.shape[0]
        random_samples_indices = np.random.choice(number_samples, number_samples, replace=True)
        return data[random_samples_indices], data_target[random_samples_indices]

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
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(data))
        predictions = np.array(predictions)

        tree_predictions = np.swapaxes(predictions, 0, 1)

        majority_predictions = []
        for prediction in tree_predictions:
            majority_predictions.append(self.determine_most_common_target_label(prediction))
        majority_predictions = np.array(majority_predictions)

        return majority_predictions
