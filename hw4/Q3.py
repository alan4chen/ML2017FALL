import sys

from RNN_w2v import Model as RNNModel
from DNN_BOW import Model as BOWModel
from config import *
from rwData import *

if __name__ == "__main__":
    X_q3 = readQ3('./data/Q3.txt')

    model = RNNModel()
    print("--load model--")
    model.loadModel("w2v120216_224_0.824100")
    X_w2v = model.process_w2v(X_q3)
    result = model.predict(X_w2v)
    print("---print RNN result---")
    print(result)

    model = BOWModel()
    print("--load model--")
    model.loadTokenizer()
    model.loadModel("bow120418_01_0.795100")
    result = model.predict(X_q3)

    print("---print BOW result--")
    print(result)



