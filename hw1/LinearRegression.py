import numpy as np
import _pickle as cPickle
from csv_train_converter import Item2ID
import csv_test_converter
import random

random.seed(0)

class Model:
	def __init__(self, weight = None, regu = 0):
		self.weight = weight
		self.regu = regu
		self.distance_nodes = []
		self.pca = None

	def compute_error(self, Y, y):
		totalError = ((Y - y) ** 2).sum()
		return (totalError / float(len(y))) ** 0.5

	def regu_error(self, Y, y):
		return self.compute_error(Y, y) + sum(self.weight[1:] ** 2) * self.regu

	def grad(self, X, y):
		difference = np.matmul(X, self.weight) - y
		summe = np.matmul(np.transpose(difference), X)
		return summe

	def step_gradient(self, X, y, learningRate):
		grad = self.grad(X, y)
		weight0 = self.weight[0] - learningRate * grad[0]
		self.weight -= learningRate * (grad + self.regu * self.weight)
		self.weight[0] = weight0


	def train(self, X, y, learning_rate = 0.00000001, num_iterations=100001):
		if type(self.weight) is not np.ndarray:
			self.weight = np.zeros(X.shape[1] + 1)
		assert len(self.weight) == X.shape[1] + 1

		Ones = np.ones(X.shape[0])
		X = np.concatenate((Ones.reshape((-1, 1)), X), axis = 1)

		for i in range(num_iterations):
			self.step_gradient(X, y, learning_rate)
			if i % 1000 == 0:
				print("Error Now: ", self.compute_error(self.predict(X), y), ", at iteration = ", i)
				print("Regu Error Now: ", self.regu_error(self.predict(X), y), ", at iteration = ", i)

	def test(self, X):
		Ones = np.ones(X.shape[0])
		X = np.concatenate((Ones.reshape((-1, 1)), X), axis=1)
		return self.predict(X)

	def predict(self, X):
		return np.matmul(X, self.weight)

	def split_validation(self, X, y, p):
		np.random.seed(0)
		length = len(X)
		randomize = np.arange(length)
		np.random.shuffle(randomize)
		X_r = X[randomize]
		y_r = y[randomize]
		train_x = X_r[:int(length * p), :]
		train_y = y_r[:int(length * p)]
		val_x = X_r[int(length * p)+1:, :]
		val_y = y_r[int(length * p)+1:]
		return train_x, train_y, val_x, val_y

	def X_transform(self, xs, train=False):
		x1 = xs[Item2ID['PM2.5']][:, :]
		x = np.concatenate((x1, x1.clip(min=0)**0.5), axis=1)
		return x



