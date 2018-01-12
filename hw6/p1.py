import argparse

import numpy as np
from skimage import io
import os

import matplotlib.pyplot as plt

def read_specific_image(dir_path, file_name):
    image = io.imread(open(os.path.join(dir_path, str(file_name)), 'rb'))
    image = image.flatten()
    return np.array(image)


def read_Image(dir_path='data/Aberdeen'):
    images = []

    file_list = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.jpg'):
            file_list.append(int(file_name.replace('.jpg','')))

    for file_name in sorted(file_list):
        # print(file_name)
        image = io.imread(open(os.path.join(dir_path, str(file_name)+'.jpg'), 'rb'))
        image = image.flatten()
        images.append(image)
    return np.array(images)

U = None
S = None
V = None
mean = None
def calSVD(images):
    global U, S, V, mean
    if U is None:
        mean = np.mean(images, axis=0)
        image_mean = images - mean
        U, S, V = np.linalg.svd(image_mean.transpose(), full_matrices=False)
    return U, S, V, mean


def average_face(images):
    mean_image = np.mean(images, axis=0)
    mean_image = mean_image.reshape((600,600,3))
    mean_image = mean_image.astype(np.uint8)
    io.imsave('p1_1_averageface.png', mean_image)


def eigenface(images, top=4):
    U, S, V, mean = calSVD(images)

    print(U.shape)
    print(S.shape)
    print(V.shape)

    for i in range(top):
        M = U[:, i]
        M -= np.min(M)
        M /= np.max(M)
        M = (M * 255).astype(np.uint8)
        io.imsave('p1_2_eigenface_'+str(i)+'.png', M.reshape((600,600,3)))

def eigenface_reconstruct(images, i, top=4): # Top 4 eigenvector
    U, S, V, mean = calSVD(images)

    weights = np.dot(images[i, :], U[:, :top])
    recon = mean + np.dot(weights, U[:, :top].transpose())

    recon -= np.min(recon)
    recon /= np.max(recon)
    recon = (recon * 255).astype(np.uint8)

    io.imsave('p1_3_recon_top' + str(top) + 'eigen_' + str(i) + '.png', recon.reshape((600, 600, 3)))

def eigenface_reconstruct_with_image(images, image, top=4): # Top 4 eigenvector
    U, S, V, mean = calSVD(images)

    weights = np.dot(image, U[:, :top])
    recon = mean + np.dot(weights, U[:, :top].transpose())

    recon -= np.min(recon)
    recon /= np.max(recon)
    recon = (recon * 255).astype(np.uint8)

    io.imsave('reconstruction.jpg', recon.reshape((600, 600, 3)))


def calculate_ratio(images, top=4):
    U, S, V, mean = calSVD(images)

    # print(S)
    print(U.shape)
    print(S.shape)
    print(V.shape)
    for i in range(top):
        print(i, 'st eigenvector ratio: ', float(S[i]) / float(sum(S)))



if __name__ == '__main__':
    import sys

    Images = read_Image(dir_path=sys.argv[1])
    image = read_specific_image(dir_path=sys.argv[1], file_name=sys.argv[2])
    eigenface_reconstruct_with_image(Images, image)

    # 1_1
    # print('cal average_face')
    # average_face(Images)

    # 1_2
    # print('cal eigenface')
    # eigenface(Images)

    # 1_3
    # print('cal eigenface recon first 10')
    # for i in range(10):
    #     eigenface_reconstruct(Images, i, top=4)

    # 1_4
    # print('cal ratio')
    # calculate_ratio(Images)

