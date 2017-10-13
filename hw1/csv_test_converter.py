
import csv
import numpy as np

from csv_train_converter import *


def path_convert(path = './test.csv'):
	tmp_data = []
	for i in range(len(ITEM_LIST)):
		tmp_data.append([])

	with open(path, 'r', encoding='utf-8') as f:
		rd = csv.reader(f, delimiter=',')
		column_num = -1
		for row in rd:
			# print(row)

			column_num += 1
			item_id = column_num % len(ITEM_LIST)
			assert row[1] == ITEM_LIST[item_id]

			tmp_row = []
			for val in row[2:]:
				if val == 'NR':
					tmp_row.append(float(0.0))
				else:
					tmp_row.append(float(val))
			tmp_data[item_id].append(tmp_row)

	testing_data = []
	for data in tmp_data:
		testing_data.append(np.array(data))

	return testing_data

def write_ans(ans, path = './ans.csv'):
	f = open(path, 'w')
	f.write('id,value\r')
	for index, val in enumerate(ans):
		f.write('id_'+str(index)+','+str(val)+'\r')
	f.flush()

if __name__ == "__main__":
	# ret = path_convert()
	# for item in ret:
	# 	print(item.shape)
	testing_data = path_convert()
	predict = testing_data[Item2ID['PM2.5']][:, -1]
	print(predict)
	print(predict.shape)
	ret = predict * 2 + 1
	print(ret)
	write_ans(ret)
