import sys

from cnn import Model
from config import *
import csv_converter


if __name__ == "__main__":
    X, y = csv_converter.readTrainData(sys.argv[1])
    model = Model()
    model.init()
    model.train(X, y)
    model.saveModel(name = VERSION_NAME, dir_path= MODEL_SAVE_DIR)
    # model.saveHistory(name = VERSION_NAME, dir_path= HISTORY_SAVE_DIR)
