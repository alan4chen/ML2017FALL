import numpy as np
from keras import Input
from keras.callbacks import ModelCheckpoint
from keras.engine import Model
from keras.layers import Dense

from p3_DataProcessor import *
from sklearn.cluster import *

import os


VERSION_NAME = "auto_encode_011002_diff-2_epoch2403"

class Kmeans:

    def __init__(self):
        self.model = None

    def fit_clustering(self, images):
        """
        :param images:
        :return: fit labels
        """
        self.model = KMeans(n_clusters=2, random_state=0).fit(images)
        return self.model.labels_

class Auto_encoder:


    def __init__(self):
        self.EPOCHS = 3000
        self.BATCH_SIZE = 128

    def build_model(self):

        # 800 epochs: private 0.69050  public 0.69184
        # encoded = Dense(128, activation="relu")(input_img)
        # encoded = Dense(64, activation="relu")(encoded)
        # encoded = Dense(32, activation="relu")(encoded)
        #
        # decoded = Dense(64, activation="relu")(encoded)
        # decoded = Dense(128, activation="relu")(decoded)
        # decoded = Dense(784, activation="sigmoid")(decoded)


        input_img = Input(shape=(784,))

        # 2400 epochs: 0.99914   0.99940

        encoded = Dense(256, activation="relu")(input_img)
        encoded = Dense(128, activation="relu")(encoded)
        encoded = Dense(64, activation="relu")(encoded)

        decoded = Dense(128, activation="relu")(encoded)
        decoded = Dense(256, activation="relu")(decoded)
        decoded = Dense(784, activation="sigmoid")(decoded)

        self.model = Model(input_img, decoded)
        self.encoder = Model(input_img, encoded)

        self.model.compile(optimizer='adam', loss='binary_crossentropy')
        self.model.summary()

    def train(self, X_train, Y_train):

        # pre train 2000 epochs
        self.model.fit(
            X_train, Y_train,
            epochs=1500,
            batch_size=self.BATCH_SIZE,
            verbose=1,
        )


        for epoch in range(1500, self.EPOCHS):

            print("EPOCHS: ", epoch, "/", self.EPOCHS)

            self.model.fit(
                X_train, Y_train,
                epochs=1,
                batch_size=self.BATCH_SIZE,
                verbose=1,
            )

            X_train_encoded = self.encoder.predict(X_train)
            kmeans_model = Kmeans()
            labels = kmeans_model.fit_clustering(X_train_encoded)

            num0 = np.count_nonzero(labels == 0)
            num1 = np.count_nonzero(labels == 1)

            print(num0, " : ", num1)

            if(abs(num0 - num1) < 100):
                print("---> save model ---->")
                self.saveModel(note="_diff"+str(num0-num1)+"_epoch"+str(epoch))


    def encoder_encode(self, X):
        return self.encoder.predict(X, verbose=1)

    def saveModel(self, model_name=VERSION_NAME, note = ""):
        self.model.save(os.path.join('./model', model_name + note +'.h5'))
        self.encoder.save(os.path.join('./encoder', model_name + note +'.h5'))

    def loadModel(self, model_name=VERSION_NAME):
        from keras.models import load_model
        self.model = load_model(os.path.join('./model', model_name + '.h5'))
        self.encoder = load_model(os.path.join('./encoder', model_name + '.h5'))


if __name__ == "__main__":

    # Argument Parse
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--test_case_path', type=str)
    parser.add_argument('--prediction_file_path', type=str)
    args = parser.parse_args()

    # Load Training Data
    print("===== load Training Image =====")
    images = load_image(path=args.image_path)

    X_train = images.astype("float32") / 255.


    if args.train:

        Y_train = images.astype("float32") / 255.

        # Train auto_encoder
        print("===== Train Auto Encoder =====")
        model = Auto_encoder()
        model.build_model()
        model.train(X_train, Y_train)

        model.saveModel()

    elif args.test:

        # encode X_train
        print("===== Encode images =====")
        model = Auto_encoder()
        model.loadModel()
        X_train_encoded = model.encoder_encode(X_train)

        # Model
        print("===== Train Clustering =====")
        model = Kmeans()
        labels = model.fit_clustering(X_train_encoded)


        # Load Testing Cases
        print("===== Load Test Cases =====")
        ans = []
        test_cases = load_test_cases(path=args.test_case_path)
        for test_case in test_cases:
            if labels[test_case[0]] == labels[test_case[1]]:
                ans.append(1)
            else:
                ans.append(0)

        print("===== Write Ans =====")
        write_ans(ans, path=args.prediction_file_path)


