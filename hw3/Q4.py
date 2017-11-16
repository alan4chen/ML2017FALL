from sys import argv
import numpy as np
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import load_model
import keras.backend as K

from csv_converter import readTrainNpy
import config

READ_FROM_NPZ = 1
CATEGORY = 7
SHAPE = 48

def main():

    print("read train data...")
    X, y = readTrainNpy()

    X = X / 255
    X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)

    print("load model...")
    model_name = "./model/" + config.VERSION_NAME + ".h5"

    label = ["angry", "disgust", "fear", "happy", "sad", "suprise", "neutral"]


    print("print saliency map...")


    emotion_classifier = load_model(model_name)
    input_img = emotion_classifier.input
    img_ids = [(0, 0), (1, 299), (2, 45), (3, 46), (4, 42), (5, 55), (6, 60)]

    for i, idx in img_ids:
        print("plot figure %d." % idx)
        plt.figure(figsize=(8, 3))

        img = X[idx].reshape(1, 48, 48, 1)

        val_proba = emotion_classifier.predict(img)
        pred = val_proba.argmax(axis=-1)
        target = K.mean(emotion_classifier.output[:, pred])
        grads = K.gradients(target, input_img)[0]
        fn = K.function([input_img, K.learning_phase()], [grads])

        heatmap = fn([img, 0])[0]
        heatmap = heatmap.reshape(48, 48)
        heatmap /= heatmap.std()

        see = img.reshape(48, 48)
        plt.title("%d. %s" % (idx, label[y[idx].argmax()]), loc='left')

        plt.subplot(1, 3, 1)
        plt.imshow(see, cmap='gray')

        thres = heatmap.std()
        see[np.where(abs(heatmap) <= thres * 1.2)] = np.mean(see)

        plt.subplot(1, 3, 2)
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar()
        plt.tight_layout()

        plt.subplot(1, 3, 3)
        plt.imshow(see, cmap='gray')
        plt.colorbar()
        plt.tight_layout()

        fig = plt.gcf()
        plt.draw()
        fig.savefig(os.path.join('./plot', '{}.png'.format(i)), dpi=100)


if __name__ == "__main__":
    main()
