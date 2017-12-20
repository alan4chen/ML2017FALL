import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open('history/Q1.pickle', 'rb')
    history = cPickle.load(f)
    fn = open('history/Q1_withoutNorm.pickle', 'rb')
    history_n = cPickle.load(fn)

    print(history.keys())
    print(history_n)

    plt.plot(history['rmse'][:100], label='training data / with Norm', c='r', linestyle='-')
    plt.plot(history['val_rmse'][:100], label='validation data / with Norm', c='r', linestyle=':')
    plt.plot(history_n['rmse'][:100], label='training data / without Norm', c='g', linestyle='-')
    plt.plot(history_n['val_rmse'][:100], label='validation data / without Norm', c='g', linestyle=':')
    plt.ylabel('rmse')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.8), loc=1)
    plt.show()


    print(history['rmse'][50], history['rmse'][100], history['rmse'][499])
    print(history['val_rmse'][50], history['val_rmse'][100], history['val_rmse'][499])
    print(history_n['rmse'][50], history_n['rmse'][100], history_n['rmse'][499])
    print(history_n['val_rmse'][50], history_n['val_rmse'][100], history_n['val_rmse'][499])