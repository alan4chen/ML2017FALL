import numpy as np
import csv
from csv_X_converter import *




def readYdata(path):
    tmp = []
    # Read Data
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter=',')
        for line in rd:
            if line[0] == "label":
                continue
            else:
                tmp.append(float(line[0]))
    Y = np.array(tmp)
    return Y

def write_ans(path, ans):
    f = open(path, 'w')
    f.write('id,label\r')
    for index, val in enumerate(ans):
        f.write(str(index+1) + ',' + str(val) + '\n')
    f.flush()

if __name__ == "__main__":

    y_train = readYdata("./Y_train")
    np.save("./y_train.npy", y_train)


