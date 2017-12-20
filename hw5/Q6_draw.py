import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    f = open('history/Q2_latentdim100.pickle', 'rb')
    history = cPickle.load(f)
    fn = open('history/Q6_latentdim100.pickle', 'rb')
    history_n = cPickle.load(fn)

    print(history.keys())
    print(history_n)

    plt.title('MF latent_dim=100 with norm')
    plt.plot(history['rmse'][:200], label='training data / origin', c='r', linestyle='-')
    plt.plot(history['val_rmse'][:200], label='validation data / origin', c='r', linestyle=':')
    plt.plot(history_n['rmse'][:200], label='training data / with gender, age, occupation', c='g', linestyle='-')
    plt.plot(history_n['val_rmse'][:200], label='validation data / with gender, age, occupation', c='g', linestyle=':')
    plt.ylabel('rmse')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.8), loc=1)
    plt.show()


    print(history['rmse'][50], history['rmse'][100], history['rmse'][499])
    print(history['val_rmse'][50], history['val_rmse'][100], history['val_rmse'][499])
    print(history_n['rmse'][50], history_n['rmse'][100], history_n['rmse'][499])
    print(history_n['val_rmse'][50], history_n['val_rmse'][100], history_n['val_rmse'][499])