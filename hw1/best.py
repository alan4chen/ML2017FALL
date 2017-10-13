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
	x2 = xs[Item2ID['O3']][:, :]
	x3 = xs[Item2ID['SO2']][:, :]
	x = np.concatenate((x1, x1.clip(min=0) ** 0.5, x2, x3), axis=1)
	return x

def myweight():
	weight = np.array([0.736335297993, -0.0384447551627, -0.00859766226533, 0.226871219135, -0.212909120058, -0.053918302016, 0.488933671026, -0.555745735797, 0.0215808997962, 1.05986860135, 0.10546261729, -0.0328556507693, -0.174101132439, -0.0954027608438, 0.152726001384, 0.100003058137, 0.0772823758212, -0.0525229418427, -0.268586761495, -0.0164955744496, 0.0265284278399, -0.0131924666276, -0.00385667475636, 0.000243227302175, -0.0214501287797, -0.00765808986142, -0.0107870188985, 0.0711081065539, -0.288663071585, 0.2149643965, -0.0350832718378, -0.122478819648, 0.0159704702284, 0.0160052066473, -0.154927006969, 0.11701870715, 0.532756543042])
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

