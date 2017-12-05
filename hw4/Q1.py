import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open('history/w2v120317.pickle', 'rb')
    history = cPickle.load(f)

    print(history.keys())

    plt.plot(history['acc'], label='training_data')
    plt.plot(history['val_acc'], label='validation_data')
    plt.ylabel('acc')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.2), loc=1)
    plt.show()