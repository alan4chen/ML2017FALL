import sys

from cnn import Model
from config import *
import csv_converter

if __name__ == "__main__":
    model = Model()
    X_test = csv_converter.readTestData(sys.argv[1])

    print("--load model--")
    model.loadModel(name = VERSION_NAME, dir_path= MODEL_SAVE_DIR)
    result = model.predict(X_test)

    # print("---print result[0:5]---")
    # print(result[0:5])

    csv_converter.writeAnsPath(result, filename=sys.argv[2])
