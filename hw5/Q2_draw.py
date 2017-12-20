import _pickle as cPickle
import os
import matplotlib.pyplot as plt

if __name__ == "__main__":
    history_dim10 = cPickle.load(open("history/Q2_latentdim10.pickle", 'rb'))
    history_dim20 = cPickle.load(open("history/Q2_latentdim20.pickle", 'rb'))
    history_dim40 = cPickle.load(open("history/Q2_latentdim40.pickle", 'rb'))
    history_dim60 = cPickle.load(open("history/Q2_latentdim60.pickle", 'rb'))
    history_dim80 = cPickle.load(open("history/Q2_latentdim80.pickle", 'rb'))
    history_dim100 = cPickle.load(open("history/Q2_latentdim100.pickle", 'rb'))

    plt.title("Training data rmse")

    plt.plot(history_dim10['rmse'][:100], label='latent_dim = 10', c='r', linestyle='-')
    plt.plot(history_dim20['rmse'][:100], label='latent_dim = 20', c='orange', linestyle='-')
    plt.plot(history_dim40['rmse'][:100], label='latent_dim = 40', c='y', linestyle='-')
    plt.plot(history_dim60['rmse'][:100], label='latent_dim = 60', c='g', linestyle='-')
    plt.plot(history_dim80['rmse'][:100], label='latent_dim = 80', c='cyan', linestyle='-')
    plt.plot(history_dim100['rmse'][:100], label='latent_dim = 100', c='b', linestyle='-')
    plt.ylabel('rmse')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.95), loc=1)
    # plt.show()


    plt.figure()
    plt.title("Validation data rmse")

    plt.plot(history_dim10['val_rmse'][:100], label='latent_dim = 10', c='r', linestyle=':')
    plt.plot(history_dim20['val_rmse'][:100], label='latent_dim = 20', c='orange', linestyle=':')
    plt.plot(history_dim40['val_rmse'][:100], label='latent_dim = 40', c='y', linestyle=':')
    plt.plot(history_dim60['val_rmse'][:100], label='latent_dim = 60', c='g', linestyle=':')
    plt.plot(history_dim80['val_rmse'][:100], label='latent_dim = 80', c='cyan', linestyle=':')
    plt.plot(history_dim100['val_rmse'][:100], label='latent_dim = 100', c='b', linestyle=':')
    plt.ylabel('rmse')
    plt.xlabel('Number of Iteration(Epoch)')
    plt.legend(bbox_to_anchor=(0.9, 0.95), loc=1)
    # plt.show()

    print(history_dim10['rmse'][100], history_dim10['val_rmse'][100])
    print(history_dim20['rmse'][100], history_dim20['val_rmse'][100])
    print(history_dim40['rmse'][100], history_dim40['val_rmse'][100])
    print(history_dim60['rmse'][100], history_dim60['val_rmse'][100])
    print(history_dim80['rmse'][100], history_dim80['val_rmse'][100])
    print(history_dim100['rmse'][100], history_dim100['val_rmse'][100])
