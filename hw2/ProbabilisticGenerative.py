
import numpy as np
from numpy.linalg import det, pinv
from math import pi

np.set_printoptions(precision = 3, suppress = True)
np.seterr(divide='ignore', invalid='ignore')

class Model:
    def __init__(self):
        self.P_C1 = 0.5
        self.P_C0 = 0.5
        self.d = 0
        self.mean_C1 = None
        self.mean_C0 = None
        self.cov_C1 = None
        self.cov_C0 = None

    def train(self, train_X, train_y):
        """ Compute Maximum Likelihood
        C1 -> y = 1, > 50k
        C0 -> y = 0, < 50k
        :param train_X: numpy array
        :param train_y: numpy array with label
        :return:
        """

        # Calculate P_C1, P_C0
        self.P_C1 = float(sum(train_y)) / float(len(train_y))
        self.P_C0 = 1.0 - self.P_C1

        # Split Array
        train_C1 = train_X[train_y == 1]
        train_C0 = train_X[train_y == 0]

        # Calculate Mean
        self.mean_C1 = train_C1.mean(axis=0)
        self.mean_C0 = train_C0.mean(axis=0)

        # Calculate Cov
        self.cov_C1 = np.cov(train_C1-self.mean_C1, rowvar=False)
        self.cov_C0 = np.cov(train_C0-self.mean_C0, rowvar=False)

        ### Calculate Cov On Hand (Not Recommended)
        # NUM_FEATURE = train_X.shape[1]
        # X_len = train_X.shape[0]
        # cov1 = np.array([[0.0 for x in range(NUM_FEATURE)] for _ in range(NUM_FEATURE)])  # cov
        # cov0 = np.array([[0.0 for x in range(NUM_FEATURE)] for _ in range(NUM_FEATURE)])  # cov
        # # covariance matrix
        # for i in range(X_len):
        #     x = train_X[i, :]
        #     x_t = x.reshape(-1, 1)
        #     if train_y[i]:
        #         cov1 += np.dot((x_t - self.mean_C1.reshape(-1,1)), [x - self.mean_C1])
        #     else:
        #         cov0 += np.dot((x_t - self.mean_C0.reshape(-1,1)), [x - self.mean_C0])
        # print(sum(train_y))
        # print(cov1)
        # print(cov1 / sum(train_y))
        # print(self.cov_C0)

        # Use Same Cov Matrix (Avoid Overfitting)
        cov = self.cov_C1 * self.P_C1 + self.cov_C0 * self.P_C0
        self.cov_C1 = cov
        self.cov_C0 = cov

        # d
        self.d = train_X.shape[1]

    def compute_P_C_X(self, X, mean, cov):
        frt = 1. / ((2. * pi) ** (self.d / 2.)) * 1. / (abs(det(cov)) ** (1./2.))
        mtx = np.sum((X - mean).dot(pinv(cov)) * (X - mean), axis = 1)
        return frt * np.exp(-1/2 * mtx)

    def predict(self, test_X):

        P_X_C0 = self.compute_P_C_X(test_X, self.mean_C0, self.cov_C0)
        P_X_C1 = self.compute_P_C_X(test_X, self.mean_C1, self.cov_C1)
        # print("true divide : ", (P_X_C1 * self.P_C1 + P_X_C0 * self.P_C0))
        probs = P_X_C1 * self.P_C1 / (P_X_C1 * self.P_C1 + P_X_C0 * self.P_C0)
        return np.array([1 if p > 0.5 else 0 for p in probs])

    def cal_accuracy(self, y, Y):
        return sum(y == Y) / len(y)

    def split_validation(self, X_all, Y_all, percentage):
        from math import log, floor
        def _shuffle(X, Y):
            np.random.seed(1)
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            return (X[randomize], Y[randomize])

        all_data_size = len(X_all)
        valid_data_size = int(floor(all_data_size * percentage))

        X_all, Y_all = _shuffle(X_all, Y_all)

        X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
        X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

        return X_train, Y_train, X_valid, Y_valid










