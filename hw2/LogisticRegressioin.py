import numpy as np
import random

random.seed(0)

class Model:
    def __init__(self, weight = None, regu = 0):
        self.weight = weight
        self.regu = regu
        self.distance_nodes = []
        self.pca = None

    def sigmoid(self, z):
        # print(z)
        return 1. / (1. + np.exp(-z))

    def step_gradient(self, X, y, learningRate):
        # print(y.shape)
        # print(self.test(X).shape)
        # print((-(y - self.test(X))).reshape(-1,1).shape)
        # print(X.shape)
        # tmp = (-(y - self.test(X)).reshape(-1, 1) * X)
        # print(tmp.shape)
        self.weight -= learningRate * np.sum((-(y - self.test(X)).reshape(-1, 1) * X), axis = 0)
        self.weight0 = self.weight[0]
        self.weight -= self.regu * self.weight * 2
        self.weight[0] = self.weight0


    def train(self, X, y, learning_rate = 0.00005, num_iterations=15001):
        if type(self.weight) is not np.ndarray:
            # Randomize weight, it's important not rand too large value
            self.weight = np.random.uniform(-1e-5, 1e-5, X.shape[1] + 1)
        assert len(self.weight) == X.shape[1] + 1

        Ones = np.ones(X.shape[0])
        X = np.concatenate((Ones.reshape((-1, 1)), X), axis = 1)

        for i in range(num_iterations):
            self.step_gradient(X, y, learning_rate)
            if i % 200 == 0:
                print("Loss Now: ", self.cal_loss(X, y), ", at iteration = ", i)

    def cal_loss(self, X, y):
        X_test = self.test(X)
        return sum(-(y * np.log(X_test) + (1. - y) * np.log(1. - X_test))) + self.regu * (sum(self.weight ** 2) - self.weight[0] ** 2)

    def predict(self, X):
        Ones = np.ones(X.shape[0])
        X = np.concatenate((Ones.reshape((-1, 1)), X), axis=1)
        tests = self.test(X)
        return np.array([1 if p > 0.5 else 0 for p in tests])

    def test(self, X):
        z = np.matmul(X, self.weight)
        Y = self.sigmoid(z)
        return Y

    def split_validation(self, X_all, Y_all, percentage):
        from math import log, floor
        random.seed(0)
        def _shuffle(X, Y):
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            return (X[randomize], Y[randomize])

        all_data_size = len(X_all)
        valid_data_size = int(floor(all_data_size * percentage))

        X_all, Y_all = _shuffle(X_all, Y_all)

        X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
        X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

        return X_train, Y_train, X_valid, Y_valid

    def cal_accuracy(self, y, Y):
        return sum(y == Y) / len(y)




