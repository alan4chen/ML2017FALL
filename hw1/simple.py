import numpy as np
import _pickle as cPickle
import random
import sys

from LinearRegression import Model
from csv_train_converter import Item2ID
import csv_test_converter

random.seed(0)




def my_transformation(xs):
	x1 = xs[Item2ID['PM2.5']][:, :]
	x = np.concatenate((x1, x1.clip(min=0)**0.5), axis=1)
	return x

def myweight():
	weight = np.array([-0.416797244384, -0.308807361039, 0.242669699853, 0.0414917780321, -0.160704448985, -0.253238253761, 0.567363256601, -0.467377606666, -0.0721095126236, 1.25296799908, 2.01786450209, -1.14325227657, 0.695431422389, -0.26921993687, 0.989871634024, 0.845106278832, -1.02623365239, -0.626974071998, -0.235828948655])
	return weight


if __name__ == "__main__":
	# Define transform function
	transform_function = my_transformation
	get_weight = myweight


	##### Load training data

	# fx = open('./training_xs.cpickle', 'rb')
	# fy = open('./training_y.cpickle', 'rb')
	# xs = cPickle.load(fx)
	# y = cPickle.load(fy)
	# fx.close()
	# fy.close()

	# x = transform_function(xs)

	# Initialize Model
	model = Model(get_weight(), regu=0.000)


	##### Train val_training data
	# train_x, train_y, val_x, val_y = model.split_validation(x, y, 0.66)
	# model.train(train_x, train_y)
	# print("model weight = ", model.weight)

	##### Predict val_testing data
	# predicted_y = model.test(val_x)
	# val_error = model.compute_error(predicted_y, val_y)
	# print("val_error = ", val_error)

	###### Train all data
	# model.train(x, y)


	testing_data = csv_test_converter.path_convert(path=sys.argv[1])
	x_test = transform_function(testing_data)
	predict = model.test(x_test)
	csv_test_converter.write_ans(predict, path=sys.argv[2])

