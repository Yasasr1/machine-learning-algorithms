import numpy as np
import pandas as pd
import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score, recall_score, plot_roc_curve

from sklearn.preprocessing import MinMaxScaler
from keras.datasets import cifar10


def load_data_breast_cancer(data_file_path):
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

    # normalize data
    # x_normalized = MinMaxScaler().fit_transform(x.values)
    # x = pd.DataFrame(x_normalized)

    # add 1 as b(intercept) value for every Xi'th row
    # x.insert(loc=len(x.columns), column='b', value=1)
    # split into training and testing sets
    x_train, x_test, y_train, y_test = tts(x, y, test_size=1/3)
    return x_train, x_test, y_train, y_test


# calculate entropy of whole dataset. (or sub dataset when called in recursive calls of build_tree)
# S = -sum(probability*log2*probability)
def calculate_entropy(x):
    entropy = 0
    # find all unique values for y(output)
    output_values = x['class'].unique()

    for value in output_values:
        probability = x['class'].value_counts()[value] / len(x['class'])
        entropy += -probability * np.log2(probability)
    return entropy


# feature entropy = sum(feature value entropy * feature value fraction) for all feature values
def calculate_feature_entropy(x, feature):
    # to overcome a divide by 0 error
    eps = np.finfo(float).eps
    feature_entropy = 0
    # find all unique values for y(output)
    output_values = x['class'].unique()
    # find all unique values for the given feature
    feature_values = x[feature].unique()

    for feature_value in feature_values:
        # entropy for each value of feature (eg - for high value of humidity feature)
        feature_value_entropy = 0
        # loop to calculate entropy of a single feature value (eg - humidity = high)
        for output_value in output_values:
            num = len(x[feature][x[feature] == feature_value][x['class'] == output_value])
            den = len(x[feature][x[feature] == feature_value])
            probability_feature_value = num / den
            feature_value_entropy += -probability_feature_value * np.log2(probability_feature_value + eps)
        fraction = den / len(x)
        feature_entropy += -fraction * feature_value_entropy
    return abs(feature_entropy)


def find_best_feature(x):
    # list to hold information gain for all features
    information_gain = []
    for feature in x.keys()[:-1]:
        information_gain.append(calculate_entropy(x) - calculate_feature_entropy(x, feature))
    return x.keys()[:-1][np.argmax(information_gain)]


def get_sub_dataset(x, node, value):
    return x[x[node] == value].reset_index(drop=True)


def build_tree(x, tree=None):
    best_feature = find_best_feature(x)
    if tree is None:
        tree = {best_feature: {}}

    # get all possible values for an attribute (humidity - low, medium, high)
    feature_values = np.unique(x[best_feature])

    for value in feature_values:
        sub_data_set = get_sub_dataset(x, best_feature, value)
        # check unique values of y and counts for current node
        y_value, count = np.unique(sub_data_set['class'], return_counts=True)
        if len(count) == 1:
            tree[best_feature][value] = y_value[0]
        else:
            tree[best_feature][value] = build_tree(sub_data_set)

    return tree


def make_prediction_id3(example, tree, default=1):
    for feature in list(example.keys()):
        if feature in list(tree.keys()):
            try:
                result = tree[feature][example[feature]]
            except:
                return default

            result = tree[feature][example[feature]]

            if isinstance(result, dict):
                return make_prediction_id3(example, result)
            else:
                return result


def id3_predict(x, tree):
    examples = x.to_dict(orient='records')
    predictions = []

    for example in examples:
        predictions.append(make_prediction_id3(example, tree, 1.0))

    return predictions


x_train, x_test, y_train, y_test = load_data_breast_cancer('./data/breast-cancer-wisconsin.data')
pd.options.mode.chained_assignment = None
x_train['class'] = y_train
id3_tree = build_tree(x_train)
pprint.pprint(id3_tree)
y_predictions = id3_predict(x_test, id3_tree)
print("accuracy on test dataset: " + str(accuracy_score(y_test.to_numpy(), y_predictions)))
# plot_roc_curve(id3_tree, x_test, y_predictions)
# plt.show()


# (trainX, trainy), (testX, testy) = cifar10.load_data()
# df = pd.DataFrame(list(zip(trainX, trainy)), columns =['Image', 'label'])
# print(df.Image)

#
#
# (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
# xVal = xTrain[49000:, :].astype(np.float)
# yVal = np.squeeze(yTrain[49000:, :])
# xTrain = xTrain[:49000, :].astype(np.float)
# yTrain = np.squeeze(yTrain[:49000, :])
# yTest = np.squeeze(yTest)
# xTest = xTest.astype(np.float)
#
# # Pre processing data
# # Normalize the data by subtract the mean image
# meanImage = np.mean(xTrain, axis=0)
# xTrain -= meanImage
# xVal -= meanImage
# xTest -= meanImage
#
# # Reshape data from channel to rows
# xTrain = np.reshape(xTrain, (xTrain.shape[0], -1))
# xVal = np.reshape(xVal, (xVal.shape[0], -1))
# xTest = np.reshape(xTest, (xTest.shape[0], -1))
#
# # Add bias dimension columns
# xTrain = np.hstack([xTrain, np.ones((xTrain.shape[0], 1))])
# xVal = np.hstack([xVal, np.ones((xVal.shape[0], 1))])
# xTest = np.hstack([xTest, np.ones((xTest.shape[0], 1))])
# print(xTrain)
