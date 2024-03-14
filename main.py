import numpy as np
import pandas as pd
from model.DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split

filename = "model/diabetes.csv"
read_data = pd.read_csv(filename)

data = read_data.iloc[:, :-1].to_numpy()
target = read_data.iloc[:, -1].to_numpy()

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=1234
)


def accuracy(target_true, target_prediction):
    return np.sum(target_true == target_prediction) / len(target_true)


decision_tree = DecisionTree()
decision_tree.train(data_train, target_train)
predictions = decision_tree.predict(data_test)

acc = accuracy(target_test, predictions)
print(acc)
