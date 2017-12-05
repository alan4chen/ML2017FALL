import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open('history/w2v120317.pickle', 'rb')
    history = cPickle.load(f)
    fn = open('history/semi120323.pickle', 'rb')
    history_n = cPickle.load(fn)

    print(history.keys())
    print(history_n)

    plt.plot(history['acc'], label='training_data')
    plt.plot(history['val_acc'], label='validation_data')
    plt.plot(history_n[0], label='training_data_with_semi')
    plt.plot(history_n[1], label='validation_data__with_semi')
    plt.ylabel('acc')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.3), loc=1)
    plt.show()