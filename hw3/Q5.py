#!/usr/bin/env python

import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras import backend as K
import config
import csv_converter


def main():
    print('load model...')
    model_name = "./model/" + config.VERSION_NAME + ".h5"
    emotion_classifier = load_model(model_name)
    layer_dict = dict([layer.name, layer] for layer in emotion_classifier.layers)

    print('load data...')
    X, y = csv_converter.readTrainNpy()


    input_img = emotion_classifier.input
    name_ls = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4', 'conv2d_5', 'conv2d_6', 'conv2d_7',
               'conv2d_8', 'conv2d_9', 'conv2d_10', 'conv2d_11', 'conv2d_12']
    collect_layers = [ K.function([input_img, K.learning_phase()], [layer_dict[name].output]) for name in name_ls ]


    # White Noise
    for cnt, fn in enumerate(collect_layers):
        photo = np.random.random((1, 48, 48, 1))
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        plt.title('{}:{}'.format(name_ls[cnt], 'random_white_noise'))
        #nb_filter = im[0].shape[3]
        nb_filter = 32
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='Blues')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('filter {}'.format(i))
            plt.tight_layout()
        img_path = os.path.join('./plot', 'filter')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path, '{}:{}'.format(name_ls[cnt], 'random_white_noise')))

    # Image299
    for cnt, fn in enumerate(collect_layers):
        X /= 255
        photo = X[299].reshape(1, 48, 48, 1)
        im = fn([photo, 0]) #get the output of that layer
        fig = plt.figure(figsize=(14, 8))
        plt.title('{}:{}'.format(name_ls[cnt], 299))
        #nb_filter = im[0].shape[3]
        nb_filter = 32
        for i in range(nb_filter):
            ax = fig.add_subplot(nb_filter/8, 8, i+1)
            ax.imshow(im[0][0, :, :, i], cmap='Blues')
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
            plt.xlabel('filter {}'.format(i))
            plt.tight_layout()
        #fig.suptitle('Output of layer{} (Given image{})'.format(cnt, choose_id))
        img_path = os.path.join('./plot', 'filter')
        if not os.path.isdir(img_path):
            os.mkdir(img_path)
        fig.savefig(os.path.join(img_path, '{}:{}'.format(name_ls[cnt], 299)))

if __name__ == '__main__':
    main()
