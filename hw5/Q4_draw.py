import _pickle as cPickle
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    f = open('history/Q1.pickle', 'rb')
    history = cPickle.load(f)
    fn = open('history/Q4_concatenate_dim500.pickle', 'rb')
    history_n = cPickle.load(fn)

    print(history.keys())
    print(history_n)

    plt.plot(history['rmse'][:100], label='training data / mf latent_dim=25', c='r', linestyle='-')
    plt.plot(history['val_rmse'][:100], label='validation data / mf latent_dim=25', c='r', linestyle=':')
    plt.plot(history_n['rmse'][:100], label='training data / DNN latent_dim=500', c='g', linestyle='-')
    plt.plot(history_n['val_rmse'][:100], label='validation data / DNN latent_dim=500', c='g', linestyle=':')
    plt.ylabel('rmse')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.95), loc=1)
    # plt.show()


    print(history['rmse'][50], history['rmse'][100], history['rmse'][499])
    print(history['val_rmse'][50], history['val_rmse'][100], history['val_rmse'][499])
    print(history_n['rmse'][50], history_n['rmse'][100], history_n['rmse'][499])
    print(history_n['val_rmse'][50], history_n['val_rmse'][100], history_n['val_rmse'][499])


    print(min(history['val_rmse']), np.argmin(history['val_rmse']))
    print(min(history_n['val_rmse']), np.argmin(history_n['val_rmse']))