import pandas as pd

from model.RandomForest import RandomForest
from sklearn.model_selection import train_test_split
from flask import Flask, jsonify, request

app = Flask(__name__)

random_forest = RandomForest(min_samples_for_split=5, max_depth=17)


def train_model():
    filename = "model/diabetes.csv"
    read_data = pd.read_csv(filename)

    data = read_data.iloc[:, :-1].to_numpy()
    target = read_data.iloc[:, -1].to_numpy()

    data_train, data_test, target_train, target_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )

    random_forest.train(data_train, target_train)


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
