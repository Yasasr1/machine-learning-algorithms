import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score


def load_data(data_file_path):
    # using breast cancer WIS consin dataset
    data = pd.read_csv(data_file_path,
                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] ,
                       names=["clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
                              "marginal_adhesion", "single_epi_cell_size", "bare_nuclei", "bland_chro",
                              "normal_nucleoli", "mitoses", "class"])
    data["class"] = data["class"].replace({2: -1, 4: 1})

    # add all features to X matrix
    # add all outputs vector Y
    x = data.iloc[:, 0:9]
    y = data.loc[:, "class"]

    # add 1 as b(intercept) value for every Xi'th row
    x.insert(loc=len(x.columns), column='b', value=1)
    # split into training and testing sets
    x_train, x_test, y_train, y_test = tts(x, y, test_size=1/3)
    return x_train, x_test, y_train, y_test


def calculate_cost(weights, x, y, reg_st):
    number_of_examples = x.shape[0]
    errors = 1 - y * (np.dot(x, weights))
    errors[errors < 0] = 0
    hinge_loss = reg_st * (np.sum(errors) / number_of_examples)
    cost = 1 / 2 * np.dot(weights, weights) + hinge_loss
    return cost


def calculate_gradient(weight_vector, x, y, reg_strength):
    error = 1 - (y * np.dot(x, weight_vector))
    if max(0, error) == 0:
        return weight_vector
    else:
        return weight_vector - (reg_strength * y * x)


def train_model(x, y, num_of_iterations):
    # creating a weights vector with zeros
    weights_vector = np.zeros(x.shape[1])
    # define regularization strength and learning rate
    c = 1500
    learning_rate = 0.000001
    # used to check if the model is converged in every 2^n th iteration
    n = 0
    cost_threshold = 0.01
    # infinite number representation in python. This is used to compare the cost value with in the initial iteration
    # because we don't know what the cost would be in the first iteration
    prev_cost = float("inf")
    for iteration in range(0, num_of_iterations):
        for index, example_value in enumerate(x):
            # pass one example at a time (SDG)
            ascent = calculate_gradient(weights_vector, example_value, y[index], c)
            weights_vector = weights_vector - (learning_rate * ascent)

        if iteration == 2 ** n or iteration == num_of_iterations - 1:
            cost = calculate_cost(weights_vector, x, y, c)
            print("Iteration: " + str(iteration) + " Cost: " + str(cost))
            if abs(prev_cost - cost) < cost_threshold * prev_cost:
                return weights_vector
            prev_cost = cost
            n += 1

    return weights_vector


x_train, x_test, y_train, y_test = load_data('./data/breast-cancer-wisconsin.data')
weights = train_model(x_train.to_numpy(), y_train.to_numpy(), 5000)
print(weights)

# testing the model on test set
y_test_predicted = []

for i in range(x_test.shape[0]):
    prediction = np.sign(np.dot(weights, x_test.to_numpy()[i]))
    y_test_predicted.append(prediction)

print("accuracy on test dataset: " + str(accuracy_score(y_test.to_numpy(), y_test_predicted)))
print("recall on test dataset: " + str(recall_score(y_test.to_numpy(), y_test_predicted)))
print("precision on test dataset: " + str(recall_score(y_test.to_numpy(), y_test_predicted)))