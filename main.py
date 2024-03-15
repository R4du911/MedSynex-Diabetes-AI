import numpy as np
import pandas as pd
from model.RandomForest import RandomForest
from sklearn.model_selection import train_test_split

filename = "model/diabetes.csv"
read_data = pd.read_csv(filename)

# read_data.info()
# print(read_data.duplicated().sum())
# print(read_data.describe())

feature_names = read_data.columns[:-1]
data = read_data.iloc[:, :-1].to_numpy()
target = read_data.iloc[:, -1].to_numpy()

data_train, data_test, target_train, target_test = train_test_split(
    data, target, test_size=0.2, random_state=1234
)


def accuracy(target_true, target_prediction):
    return np.sum(target_true == target_prediction) / len(target_true)


random_forest = RandomForest()
random_forest.train(data_train, target_train)
predictions_random_forest = random_forest.predict(data_test)

acc_random_forest = accuracy(target_test, predictions_random_forest)

print("\nAccuracy Random Forest: ")
print(acc_random_forest)
