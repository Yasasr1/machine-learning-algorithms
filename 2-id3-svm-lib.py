import six
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve
from tensorflow.keras.datasets import cifar10
sys.modules['sklearn.externals.six'] = six
from id3 import Id3Estimator


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

    # normalize data
    x_normalized = MinMaxScaler().fit_transform(x.values)
    x = pd.DataFrame(x_normalized)

    # split into training and testing sets
    x_train, x_test, y_train, y_test = tts(x, y, test_size=1/3)
    return x_train, x_test, y_train, y_test


def run_id3_cifar(x_train, y_train, x_test, y_test):
    # x_train = x_train[:20000, :].astype(float)
    # y_train = np.squeeze(y_train[:20000, :])
    
    x_train = x_train.astype(float)
    y_train = np.squeeze(y_train)
    
    
    y_test = np.squeeze(y_test)
    x_test = x_test.astype(float)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))

    clf = Id3Estimator()
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    print("ID3 with CIFAR-10 dataset")
    print("Testing accuracy:", accuracy_score(y_test, y_pred_test))
    print("Training accuracy:", accuracy_score(y_train, y_pred_train))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))



def run_id3_breast_cancer(x_train, x_test, y_train, y_test):
    clf = Id3Estimator()
    clf.fit(x_train, y_train)
    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    print("ID3 with breast cancer dataset")
    print("Testing accuracy:", accuracy_score(y_test, y_pred_test))
    print("Training accuracy:", accuracy_score(y_train, y_pred_train))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))



def run_svm_cifar(x_train, y_train, x_test, y_test):
    x_train = x_train[:1000, :].astype(float)
    y_train = np.squeeze(y_train[:1000, :])
    y_test = np.squeeze(y_test)
    x_test = x_test.astype(float)
    x_train = np.reshape(x_train, (x_train.shape[0], -1))
    x_test = np.reshape(x_test, (x_test.shape[0], -1))
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    print("Linear SVM with CIFAR-10 dataset")
    print("Testing accuracy:", accuracy_score(y_test, y_pred_test))
    print("Training accuracy:", accuracy_score(y_train, y_pred_train))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))



def run_svm_breast_cancer(x_train, x_test, y_train, y_test):
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)

    y_pred_test = clf.predict(x_test)
    y_pred_train = clf.predict(x_train)

    print("Linear SVM with breast cancer dataset")
    print("Testing accuracy:", accuracy_score(y_test, y_pred_test))
    print("Training accuracy:", accuracy_score(y_train, y_pred_train))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_test))
    print(classification_report(y_test, y_pred_test))
    plot_roc_curve(clf, x_test, y_pred_test)
    plt.show()


# load data
(xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
x_train_bc, x_test_bc, y_train_bc, y_test_bc = load_data('./data/breast-cancer-wisconsin.data')


# run classifiers
run_id3_cifar(xTrain, yTrain, xTest, yTest)
#run_id3_breast_cancer(x_train_bc, x_test_bc, y_train_bc, y_test_bc)
#run_svm_cifar(xTrain, yTrain, xTest, yTest)
#run_svm_breast_cancer(x_train_bc, x_test_bc, y_train_bc, y_test_bc)
