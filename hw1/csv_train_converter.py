import csv
import numpy as np

ITEM_LIST = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10',
			'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC',
			'WIND_SPEED', 'WS_HR']

Item2ID = dict()
for item_id, item in enumerate(ITEM_LIST):
	Item2ID[item] = item_id

if __name__ == "__main__":

	tmp_data = []
	for j in range(12):
		tmp_data.append([])
		for i in range(len(ITEM_LIST)):
			tmp_data[j].append([])

	# print(tmp_data)

	# Read Data
	with open('./train.csv', 'r', encoding='big5') as f:
		rd = csv.reader(f, delimiter=',')
		column_num = -1
		month_num = 0
		current_month = '1/'
		for row in rd:
			if row[0][0] != '2':
				continue

			column_num += 1
			if row[0][5:7] != current_month:
				month_num += 1
				current_month = row[0][5:7]
			assert ITEM_LIST[column_num % len(ITEM_LIST)] == row[2]

			for i in range(3, len(row)):
				if row[i] == 'NR':
					rs = 0
				else:
					rs = float(row[i])
				tmp_data[month_num][column_num % len(ITEM_LIST)].append(rs)

	training_data = []
	for item_id in range(len(ITEM_LIST)):
		training_data.append([])
	for month in range(12):
		for item_id in range(len(ITEM_LIST)):
			for start_hour in range(len(tmp_data[month][item_id])-9):
				training_data[item_id].append(tmp_data[month][item_id][start_hour:start_hour+9])
	print(training_data[2])

	training_data_target = []
	for month in range(12):
		for predict_hour in range(9, len(tmp_data[month][9])):
			training_data_target.append(tmp_data[month][9][predict_hour])
	print(training_data_target)


	training_xs = []
	for item_id in range(len(ITEM_LIST)):
		training_xs.append(np.array(training_data[item_id]))
		print(training_xs[item_id].shape)
	training_y = np.array(np.array(training_data_target))
	print(training_y.shape)

	import _pickle as cPickle
	f = open("./training_xs.cpickle", 'wb')
	cPickle.dump(training_xs, f)
	f.close()

	f = open("./training_y.cpickle", 'wb')
	cPickle.dump(training_y, f)
	f.close()
