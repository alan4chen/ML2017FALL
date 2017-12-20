

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

    def __init__(self, latent_dim, sigma, user_nums=0, movie_nums=0, epochs=500, ):
        self.latent_dim = latent_dim
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
        input_gender = Input(shape=(1,), name='input_gender')  # user gender
        input_age = Input(shape=(1,), name='input_age')  # user age
        input_occu = Input(shape=(1,), name='input_occu')  # user occu

        embedded_user = Embedding(self.user_nums, self.latent_dim, name='emb_userID')(input_user)
        embedded_movie = Embedding(self.movie_nums, self.latent_dim, name='emb_movieID')(input_movie)
        vector_user = Dropout(0.5)(Flatten(name='vec_userID')(embedded_user))
        vector_movie = Dropout(0.5)(Flatten(name='vec_movieID')(embedded_movie))

        dot = Dot(axes=1)([vector_user, vector_movie])

        embedded2_user = Embedding(self.user_nums, 1, name='emb2_userID')(input_user)
        embedded2_movie = Embedding(self.movie_nums, 1, name='emb2_movieID')(input_movie)
        embedded_gender = Embedding(self.movie_nums, 1, name='emb_gender')(input_gender)
        embedded_occu = Embedding(self.movie_nums, 1, name='emb_occu')(input_occu)
        dense_age = Dense(1, name='dense_age')(input_age)
        bias_user = Flatten(name='bias_user')(embedded2_user)
        bias_movie = Flatten(name='bias_movie')(embedded2_movie)
        bias_gender = Flatten(name='bias_gender')(embedded_gender)
        bias_occu = Flatten(name='bias_occu')(embedded_occu)

        outputs = Add()([bias_user, bias_movie, bias_gender, bias_occu, dense_age, dot])

        self.model = kModel(inputs=[input_user, input_movie, input_gender, input_age, input_occu], outputs=outputs)
        self.model.summary()

        self.model.compile(optimizer='adam', loss='mse', metrics=[self.rmse])

    def train(self, train_user, train_movie,  train_gender, train_age, train_occu, train_rating):

        #### Train Model
        print("-- Train Model --")

        ### Shuffle
        from math import floor
        np.random.seed(1)
        size = len(train_user)
        randomize = np.arange(size)
        np.random.shuffle(randomize)
        train_size = int(floor(size * 0.95))
        X1_train = train_user[randomize][:train_size]
        X2_train = train_movie[randomize][:train_size]
        X3_train = train_gender[randomize][:train_size]
        X4_train = train_age[randomize][:train_size]
        X5_train = train_occu[randomize][:train_size]
        Y_train = train_rating[randomize][:train_size]
        X1_valid = train_user[randomize][train_size:]
        X2_valid = train_movie[randomize][train_size:]
        X3_valid = train_gender[randomize][train_size:]
        X4_valid = train_age[randomize][train_size:]
        X5_valid = train_occu[randomize][train_size:]
        Y_valid = train_rating[randomize][train_size:]

        filepath = os.path.join('./model', config.VERSION_NAME + '_latentdim' + str(self.latent_dim)
                                + "_{epoch:02d}_{val_rmse:.6f}.h5")
        cp = ModelCheckpoint(monitor='val_rmse', save_best_only=True,
                             mode='min', filepath=filepath, verbose=1)


        self.history = self.model.fit([X1_train, X2_train, X3_train, X4_train, X5_train], Y_train,
                            epochs=self.EPOCHS, verbose=1, batch_size=8192, callbacks=[cp],
                            validation_data=([X1_valid, X2_valid, X3_valid, X4_valid, X5_valid], Y_valid))

    def test(self, test_user, test_movie, test_gender, test_age, test_occu):
        return self.model.predict([test_user, test_movie, test_gender, test_age, test_occu])


    def saveHistory(self, name=config.VERSION_NAME, dir_path='./history'):
        # print(self.history.history['acc'])

        fw = open(os.path.join(dir_path, name + '_latentdim' + str(self.latent_dim) + '.pickle'), 'wb')
        cPickle.dump(self.history.history, fw)

    def saveModel(self, model_name=config.VERSION_NAME):
        self.model.save(os.path.join('./model', model_name + '_latentdim' + str(self.latent_dim) + '.h5'))

    def loadModel(self, model_name=config.VERSION_NAME):
        from keras.models import load_model
        self.model = load_model(os.path.join('./model', model_name + '.h5'),
                                custom_objects={'rmse': self.rmse})


def user2feature(userIDs, userID_np, gender_np, age_np, occu_np):
    gender_list = []
    age_list = []
    occu_list = []

    for id in userIDs:
        gender_list.append(gender_np[np.where(userID_np==id)])
        age_list.append(age_np[np.where(userID_np == id)])
        occu_list.append(occu_np[np.where(userID_np == id)])
    return np.array(gender_list), np.array(age_list), np.array(occu_list)



import csv_reader

if __name__ == "__main__":

    # Argument Parse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_csv_path', type=str)
    parser.add_argument('--train_csv_path', type=str)
    parser.add_argument('--prediction_file_path', type=str)
    parser.add_argument('--movies_csv_path', type=str)
    parser.add_argument('--users_csv_path', type=str)
    args = parser.parse_args()

    if args.train:

        ldim = 100
        print("======  TRAIN Latent DIM: ", ldim)

        userID_np, gender_np, age_np, occu_np = csv_reader.readUsers(path=args.users_csv_path)
        movieID_np = csv_reader.readMovies(path=args.movies_csv_path)
        train_userID, train_movieID, train_rating = csv_reader.readTrain(path=args.train_csv_path)
        train_gender, train_age, train_occu = user2feature(train_userID, userID_np, gender_np, age_np, occu_np)

        # Normalize
        mu = np.mean(train_rating)
        sigma = np.std(train_rating)
        cPickle.dump((mu, sigma), open("Q6.para", 'wb'))
        train_rating = (train_rating - mu) / sigma

        model = Model(ldim, sigma, max(userID_np)+1, max(movieID_np)+1)
        model.build()


        model.train(train_userID, train_movieID, train_gender, train_age, train_occu, train_rating)

        model.saveModel()
        model.saveHistory()

    elif args.test:



        # Load normalized paras
        mu, sigma = cPickle.load(open("Q6.para", 'rb'))

        userID_np, gender_np, age_np, occu_np = csv_reader.readUsers(path=args.users_csv_path)
        movieID_np = csv_reader.readMovies(path=args.movies_csv_path)
        test_userID, test_movieID = csv_reader.readTest(path=args.test_csv_path)
        test_gender, test_age, test_occu = user2feature(test_userID, userID_np, gender_np, age_np, occu_np)

        print("matrix shape:")
        for matrix in (test_userID, test_movieID, test_gender, test_age, test_occu):
            print(matrix.shape)


        model = Model(0, sigma)
        model.loadModel()

        ans = model.test(test_userID, test_movieID, test_gender, test_age, test_occu)
        # de Normalize
        ans = ans * sigma + mu

        csv_reader.writeAns(ans, filepath=args.prediction_file_path)