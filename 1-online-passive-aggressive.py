import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler


def load_data(data_file_path):
    # using breast cancer WIS consin dataset
    data = pd.read_csv(data_file_path,
                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       names=["clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
                              "marginal_adhesion", "single_epi_cell_size", "bare_nuclei", "bland_chro",
                              "normal_nucleoli", "mitoses", "class"])
    data["class"] = data["class"].replace({2: -1, 4: 1})

    # add all features to X matrix
    # add all outputs vector Y
    x = data.iloc[:, 0:9]
    y = data.loc[:, "class"]

    # normalize data
    x_normalized = MinMaxScaler().fit_transform(x.values)
    x = pd.DataFrame(x_normalized)

    # add 1 as b(intercept) value for every Xi'th row
    x.insert(loc=len(x.columns), column='b', value=1)
    # split into training and testing sets
    x_train, x_test, y_train, y_test = tts(x, y, test_size=1 / 3)
    return x_train, x_test, y_train, y_test


def calculate_loss(w, x_instance, y_instance):
    return max(0, (1 - y_instance * np.dot(w, x_instance)))


def calculate_update_pa_ii(loss, x_instance, c):
    aggressiveness_parameter_term = 1 / 2 * c
    update_value = loss / (np.dot(x_instance, x_instance) + aggressiveness_parameter_term)
    return update_value


def train_model(x, y, num_of_iterations):
    # create a weigth vector
    w = np.zeros(x.shape[1])
    w_1 = np.zeros(x.shape[1])
    w_2 = np.zeros(x.shape[1])
    w_10 = np.zeros(x.shape[1])
    # define C(aggressiveness parameter)
    c = 1

    for iteration in range(0, num_of_iterations):
        for index, example_value in enumerate(x):
            # pass one example at a time to train the model
            loss = calculate_loss(w, example_value, y[index])
            update = calculate_update_pa_ii(loss, example_value, c)
            w = w + (update * y[index] * example_value)

        if iteration == 1:
            w_1 = w
        if iteration == 2:
            w_2 = w
        if iteration == 9:
            w_10 = w

    return w_1, w_2, w_10


def predict_category(w_1, w_2, w_10, x):
    y_predicted_1 = []
    y_predicted_2 = []
    y_predicted_10 = []

    for i in range(x.shape[0]):
        y_predicted_1.append(np.sign(np.dot(w_1, x.to_numpy()[i])))
        y_predicted_2.append(np.sign(np.dot(w_2, x.to_numpy()[i])))
        y_predicted_10.append(np.sign(np.dot(w_10, x.to_numpy()[i])))

    return y_predicted_1, y_predicted_2, y_predicted_10


x_train, x_test, y_train, y_test = load_data('./data/breast-cancer-wisconsin.data')
w_1, w_2, w_10 = train_model(x_train.to_numpy(), y_train.to_numpy(), 10)

y_test_predicted_1, y_test_predicted_2, y_test_predicted_10 = predict_category(w_1, w_2, w_10, x_test)
y_train_predicted_1, y_train_predicted_2, y_train_predicted_10 = predict_category(w_1, w_2, w_10, x_train)

print("testing accuracy on 1 iteration: " + str(accuracy_score(y_test.to_numpy(), y_test_predicted_1)) +
      " training accuracy on 1 iteration: " + str(accuracy_score(y_train.to_numpy(), y_train_predicted_1)))

print("accuracy on 2 iterations: " + str(accuracy_score(y_test.to_numpy(), y_test_predicted_2)) +
      " training accuracy on 2 iteration: " + str(accuracy_score(y_train.to_numpy(), y_train_predicted_2)))

print("accuracy on 10 iterations: " + str(accuracy_score(y_test.to_numpy(), y_test_predicted_10)) +
      " training accuracy on 10 iteration: " + str(accuracy_score(y_train.to_numpy(), y_train_predicted_10)))
