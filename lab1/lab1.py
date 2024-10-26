import numpy as np
from sklearn.model_selection import RepeatedKFold # cross validation
from sklearn.naive_bayes import GaussianNB # gaussian naive bayes classifier
from sklearn.neighbors import KNeighborsClassifier # nearest neighbors classifier
from sklearn.metrics import accuracy_score
import pandas as pd
import random
from datetime import datetime
import os
import threading


def read_dataset_names():
    dataset_names = os.listdir("../datasets")
    prefix = "../datasets/"
    datasets = []
    for i in range(30):
        file_name = prefix+dataset_names[i]
        datasets.append(file_name)
    print(datasets)
    return datasets


def run_experiment(file_name):
    df = pd.read_csv(file_name, sep=",", header=None)
    x = df.values
    print(df.shape)
    y = np.random.randint(0, 2, size=x.shape[0])

    gnb = GaussianNB()
    knn = KNeighborsClassifier()
    rkf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=random.seed(datetime.now().timestamp()))

    results_matrix = np.zeros((2, 10))
    rkf.get_n_splits(x, y)

    for i, (train_index, test_index) in enumerate(rkf.split(x)):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        #   gaussian naibe bayes 
        gnb.fit(x_train, y_train)
        y_gnb_predict = gnb.predict(x_test)
        gnb_accuracy = accuracy_score(y_test, y_gnb_predict)
        results_matrix[0, i] = gnb_accuracy

        #   knn
        knn.fit(x_train, y_train)
        y_knn_predict = knn.predict(x_test)
        knn_accuracy = accuracy_score(y_test, y_knn_predict)
        results_matrix[1, i] = knn_accuracy

    result = pd.DataFrame(results_matrix)
    print("\n - - - - - - - - - - - - - - - - - - - - - - - - \n")
    print(file_name)
    print(result.transpose())
    return result


# def threading_experiment():
#     threads = []
#     for i in range(30):
#         t = threading.Thread(target=)


if __name__ == "__main__":
    names = read_dataset_names()
    for file_name in names:
        run_experiment(file_name)
    