

import keras.backend as K
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers.merge import Dot, Add, Concatenate
from keras.models import Model as kModel
from keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np
import os
import _pickle as cPickle

import config



class Model:

    def __init__(self, sigma, user_nums=0, movie_nums=0, epochs=500, ):
        self.sigma = sigma

        self.user_nums = user_nums
        self.movie_nums = movie_nums

        self.model = None
        self.history = None

        self.EPOCHS = epochs

    def rmse(self, y_true, y_pred):
        return K.sqrt( K.mean((y_pred * self.sigma - y_true * self.sigma)**2) )


    def build(self):
        print("-- Build Model --")

        input_user = Input(shape=(1,), name='in_userID')  # user id
        input_movie = Input(shape=(1,), name='in_movieID')  # movie id

        embedded_user = Embedding(self.user_nums, 100, name='emb_userID')(input_user)
        embedded_movie = Embedding(self.movie_nums, 100, name='emb_movieID')(input_movie)

        user_vec = Flatten(name='user')(embedded_user)
        movie_vec = Flatten(name='movie')(embedded_movie)

        embedded_concate = Concatenate()([user_vec, movie_vec])
        hidden = Dense(units=20, activation='relu')(embedded_concate)
        hidden = Dense(units=10, activation='relu')(hidden)
        outputs = Dense(units=1)(hidden)

        self.model = kModel(inputs=[input_user, input_movie], outputs=outputs)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mse', metrics=[self.rmse])

    def train(self, train_user, train_movie, train_rating):

        #### Train Model
        print("-- Train Model --")

        X1_train, X2_train, Y_train, X1_valid, X2_valid, Y_valid = self.split_validation(
                    train_user, train_movie, train_rating, 0.95)

        filepath = os.path.join('./model', config.VERSION_NAME + "_{epoch:02d}_{val_rmse:.6f}.h5")
        cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True,
                             mode='min', filepath=filepath, verbose=1)



        self.history = self.model.fit([X1_train, X2_train], Y_train,
                            epochs=self.EPOCHS, verbose=1, batch_size=8192, callbacks=[cp],
                            validation_data=([X1_valid, X2_valid], Y_valid))

    def test(self, train_user, train_movie):
        return self.model.predict([train_user, train_movie])


    def saveHistory(self, name=config.VERSION_NAME, dir_path='./history'):
        # print(self.history.history['acc'])

        fw = open(os.path.join(dir_path, name + '.pickle'), 'wb')
        cPickle.dump(self.history.history, fw)

    def saveModel(self, model_name=config.VERSION_NAME):
        self.model.save(os.path.join('./model', model_name + '.h5'))

    def loadModel(self, model_name=config.VERSION_NAME):
        from keras.models import load_model
        self.model = load_model(os.path.join('./model', model_name + '.h5'), custom_objects={'rmse': self.rmse})


    def split_validation(self, X1_all, X2_all, Y_all, percentage):
        from math import log, floor
        def _shuffle(X1, X2, Y):
            np.random.seed(1)
            randomize = np.arange(len(X1))
            np.random.shuffle(randomize)
            return (X1[randomize], X2[randomize], Y[randomize])

        all_data_size = len(X1_all)
        valid_data_size = int(floor(all_data_size * percentage))

        X1_all, X2_all,  Y_all = _shuffle(X1_all, X2_all, Y_all)

        X1_train, X2_train, Y_train = X1_all[0:valid_data_size], X2_all[0:valid_data_size], Y_all[0:valid_data_size]
        X1_valid, X2_valid, Y_valid = X1_all[valid_data_size:], X2_all[valid_data_size:], Y_all[valid_data_size:]

        return X1_train, X2_train, Y_train, X1_valid, X2_valid, Y_valid


import csv_reader

if __name__ == "__main__":

    # Argument Parse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()

    if args.train:

        userID_np, gender_np, age_np, occu_np = csv_reader.readUsers()
        movieID_np = csv_reader.readMovies()
        train_userID, train_movieID, train_rating = csv_reader.readTrain()

        # Normalize
        mu = np.mean(train_rating)
        sigma = np.std(train_rating)
        cPickle.dump((mu, sigma), open("Q6.para", 'wb'))
        train_rating = (train_rating - mu) / sigma

        print("userID_np.shape: ", userID_np.shape, "  movieID_np.shape:", movieID_np.shape)
        model = Model(sigma, max(userID_np)+1, max(movieID_np)+1)
        model.build()

        model.train(train_userID, train_movieID, train_rating)

        model.saveModel()
        model.saveHistory()

    elif args.test:

        # Load normalized paras
        mu, sigma = cPickle.load(open("Q6.para", 'rb'))

        userID_np, movieID_np = csv_reader.readTest()

        model = Model(sigma)
        model.loadModel()

        ans = model.test(userID_np, movieID_np)
        # de Normalize
        ans = ans * sigma + mu

        csv_reader.writeAns(ans, filepath="./ans/" + config.VERSION_NAME + ".csv")