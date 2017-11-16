import numpy as np
import csv
import os


def readTrainData(path):
    from keras.utils import to_categorical
    features = []
    labels = []
    # Read Data
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter=',')
        for line in rd:
            if line[0] == "label":
                continue
            else:
                labels.append(line[0])
                features.append(np.array([float(_) for _ in line[1].split(" ")]))
    X = np.array(features)
    y = to_categorical(labels,7)
    return X, y

def readTestData(path="data/test.csv"):
    from keras.utils import to_categorical
    features = []
    # Read Data
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter=',')
        for line in rd:
            if line[0] == "id":
                continue
            else:
                features.append(np.array([float(_) for _ in line[1].split(" ")]))
    X = np.array(features)
    return X

def readTrainNpy(dir_path = './data'):
    return np.load(os.path.join(dir_path, 'X_train.npy')), np.load(os.path.join(dir_path, 'y_train.npy'))


def writeAns(result, filename, dir='./ans'):
    with open(os.path.join(dir, filename+".csv"), 'w') as f:
        f.write('id,label\n')
        for i in range(len(result)):
            predict = np.argmax(result[i])
            f.write(repr(i) + ',' + repr(predict) + '\n')

def writeAnsPath(result, filename):
    with open(filename, 'w') as f:
        f.write('id,label\n')
        for i in range(len(result)):
            predict = np.argmax(result[i])
            f.write(repr(i) + ',' + repr(predict) + '\n')


if __name__ == "__main__":

    X_train, y_train = readTrainNpy()
    print(X_train.shape)
    print(y_train.shape)
    print(X_train[0:10, :])
    print(y_train[0:10, :])

    # Save data if call readTrainData
    # np.save('data/X_train.npy', X_train)
    # np.save('data/y_train.npy', y_train)
