import numpy as np


def load_image(path = 'data/image.npy'):
    """
    :param path:
    :return: list of np.array (dim = 784)
    """
    return np.load(open(path, 'rb'))

def load_test_cases(path = 'data/test_case.csv'):
    """
    :param path:
    :return:    [ [imageID, imageID], [  ,  ], .... ]
    """
    f = open(path, 'r')
    test_cases = []
    for line in f.readlines()[1:]:
        splitted = line.split(',')
        test_cases.append([int(splitted[1]), int(splitted[2])])
    return test_cases


def write_ans(ans, path = 'ans/ans.csv'):
    f = open(path, 'w')
    f.write('ID,Ans\n')
    for i in range(len(ans)):
        f.write(repr(i) + ',' + repr(int(ans[i])) + '\n')
