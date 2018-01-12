
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.cluster import *

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt

from p3_autoencode import Auto_encoder


if __name__ == "__main__":
    images = np.load(open('data/visualization.npy', 'rb'))
    print(images.shape)

    print("### Load Model")
    model = Auto_encoder()
    model.loadModel()

    X_train_encoded = model.encoder_encode(images)

    print("SVD:")
    mean = np.mean(X_train_encoded, axis=0)
    image_mean = X_train_encoded - mean
    U, S, V = np.linalg.svd(image_mean.transpose(), full_matrices=False)
    vis_data = np.dot(X_train_encoded, U[:, :2])
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    print("vis_x.shape:", vis_x.shape)
    print("vis_y.shape:", vis_y.shape)

    print(vis_x[:10])
    print(vis_y[:10])

    labels = [0]*5000 + [1]*5000
    print(labels)

    plt.figure()
    plt.scatter(vis_x, vis_y, c=labels)
    import random
    for i, txt in enumerate(labels):
        if(random.random()<0.01):
            plt.annotate(str(i), (vis_x[i],vis_y[i]))
    plt.savefig("Q3_3num.png")

