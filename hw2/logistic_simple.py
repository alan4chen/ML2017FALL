import numpy as np
import sys

from csv_X_converter import *
from csv_Y_converter import *
from LogisticRegressioin import Model


def Xtransform(xs):
    x = xs
    for i in [0, 1, 3, 4, 5]:
        mean = np.mean(x[:, i], axis=0)
        std = np.std(x[:, i], axis=0)
        x[:, i] = (x[:, i] - mean) / std
    # print("X transformed shape:", x.shape)
    return x

if __name__ == "__main__":

    # Train
    X_train_all = readXdata(sys.argv[1])
    y_train_all = readYdata(sys.argv[2])

    # Init Model
    model = Model()

    # Train All Data
    Y_train_all = model.train(Xtransform(X_train_all), y_train_all)
    # acc_train = model.cal_accuracy(Y_train_all, y_train_all)
    # print("All Training Data Accuracy:", acc_train, "\n")

    # Predict
    X_test = readXdata(sys.argv[3])
    Y_test = model.predict(Xtransform(X_test))

    # print(Y_test)
    write_ans(sys.argv[4], Y_test)

