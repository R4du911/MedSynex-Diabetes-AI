import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from model.RandomForest import RandomForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from flask import Flask, jsonify, request

app = Flask(__name__)

random_forest = RandomForest(min_samples_for_split=5, max_depth=17)


def train_model():
    filename = "model/diabetes.csv"
    read_data = pd.read_csv(filename)

    data = read_data.iloc[:, :-1].to_numpy()
    target = read_data.iloc[:, -1].to_numpy()

    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=52
    )

    # random_forest_for_hyperparameter_tuning = RandomForestClassifier()
    # param_grid = [{
    #     "max_depth": [5, 10, 13, 15, 17, 20, 25, 30],
    #     "min_samples_split": [2, 5, 7, 10]
    # }]
    # random_forest_searcher = RandomizedSearchCV(estimator=random_forest_for_hyperparameter_tuning,
    #                                             param_distributions=param_grid,
    #                                             cv=5, random_state=52)
    # random_forest_searcher.fit(data_train, target_train)
    # print(random_forest_searcher.best_params_)

    # def accuracy(target_true, target_prediction):
    #     return np.sum(target_true == target_prediction) / len(target_true)

    random_forest.train(data_train, target_train)
    # predictions_random_forest = random_forest.predict(data_test)

    # acc_random_forest = accuracy(target_test, predictions_random_forest)

    # print("\nModel accuracy: ")
    # print(acc_random_forest)
    # print("\n")


@app.route("/diabetes-prediction", methods=['POST'])
def diabetes_prediction():
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data]).to_numpy()

    prediction = random_forest.predict(input_df)

    return jsonify(prediction.tolist())


if __name__ == '__main__':
    with app.app_context():
        train_model()
        app.run()
