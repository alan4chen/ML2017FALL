import sys

from RNN_w2v import Model
from config import *
from rwData import *

if __name__ == "__main__":
    model = Model()
    X_test = readTestingTXTstemed(sys.argv[1])

    print("--load model--")
    model.loadModel()
    X_w2v = model.process_w2v(X_test)
    result = model.predict(X_w2v)

    # print("---print result[0:1]---")
    # print(result[0:1])
    print("--Predict Complete--")

    writeAnsPath(result, filepath = sys.argv[2])
