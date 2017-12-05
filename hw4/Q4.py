import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open('history/w2v120317.pickle', 'rb')
    history = cPickle.load(f)
    fn = open('history/noFuhao120513.pickle', 'rb')
    history_n = cPickle.load(fn)

    print(history.keys())

    plt.plot(history['acc'], label='training_data')
    plt.plot(history['val_acc'], label='validation_data')
    plt.plot(history_n['acc'], label='training_data_without_punctuation_mark')
    plt.plot(history_n['val_acc'], label='validation_data_without_punctuation_mark')
    plt.ylabel('acc')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.3), loc=1)
    plt.show()