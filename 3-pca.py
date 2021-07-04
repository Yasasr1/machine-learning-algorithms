import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def load_data(data_file_path):
    # using breast cancer WIS consin dataset
    data = pd.read_csv(data_file_path,
                       usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       names=["clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
                              "marginal_adhesion", "single_epi_cell_size", "bare_nuclei", "bland_chro",
                              "normal_nucleoli", "mitoses", "class"])
    data["class"] = data["class"].replace({2: "benign", 4: "malignant"})

    # add all features to X matrix
    # add all outputs vector Y
    x = data.iloc[:, 0:9]
    y = data.loc[:, "class"]
    x_normalized = StandardScaler().fit_transform(x)
    return x_normalized, y


def create_covariance_matrix(x):
    covariance_matrix = np.cov(x.T)
    return covariance_matrix


def pca(covariance_matrix, x, y):
    # get eigenvectors and eigenvalues from the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    # calculate explained variance percentage for each eigenvector
    explained_variances = []
    eigenvalues_sum = np.sum(eigenvalues)
    for index in range(len(eigenvalues)):
        explained_variances.append(eigenvalues[index] / eigenvalues_sum)

    # print(explained_variances)
    # first 2 elements of the explained_variances list have highest values
    pc1 = x.dot(eigenvectors.T[0])
    pc2 = x.dot(eigenvectors.T[1])

    df = pd.DataFrame(pc1, columns=["PC1"])
    df["PC2"] = pc2
    df['y'] = y
    print(df['y'])
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="y")
    plt.show()


x, y = load_data('./data/breast-cancer-wisconsin.data')
cov_matrix = create_covariance_matrix(x)
pca(cov_matrix, 2, x, y)
