import sys

from RNN_w2v import Model
from config import *
from rwData import *


if __name__ == "__main__":
    training_label, training_text = readTrainingLabelTXTstemed(sys.argv[1])
    training_nolabel_text = readTrainingNoLabelTXTstemed(sys.argv[2])
    # testing_text = readTestingTXTstemed()

    model = Model()

    model.fit_w2v(training_text, training_nolabel_text)
    training_vec = model.process_w2v(training_text)

    model.init()
    model.train(training_vec, training_label)
    model.saveModel()
    # model.saveHistory()
