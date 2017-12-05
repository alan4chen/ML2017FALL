
import _pickle as cPickle
import os
import numpy as np
import config

from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D
from keras.layers import GRU, LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint

from keras.layers.wrappers import Bidirectional
from keras.preprocessing.image import ImageDataGenerator

from gensim.models.word2vec import Word2Vec



class Model:

    def __init__(self, seqlen = 35, STEPS_PER_EPOCH=30):
        self.tokenizer = None
        self.tokenNumWords = 10000
        self.val_score = None
        self.seqlen = seqlen
        self.w2v_features = 80

        self.BATCH = 128
        self.EPOCH = 10
        self.STEPS_PER_EPOCH = STEPS_PER_EPOCH

    def init(self):
        self.model = Sequential()
        print("Max len of tokenizer.word", self.tokenNumWords)

        self.model.add(Dense(4096, activation='relu',input_shape=(self.tokenNumWords, )))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(521, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(8, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, X, y):
        X = self.process_bow(X)

        X_train, y_train, X_val, y_val = self.split_validation(X, y, 0.9)

        filepath = os.path.join('./model', config.VERSION_NAME+"_{epoch:02d}_{val_acc:.6f}.h5")
        model_checkpoint = ModelCheckpoint(filepath, verbose=1, period=1, save_best_only=True, monitor='val_acc')

        self.history = self.model.fit(X_train, y_train, epochs=self.EPOCH, verbose=1,
                                      validation_data=(X_val, y_val), callbacks=[model_checkpoint])

        score = self.model.evaluate(X, y)
        print('Train accuracy (all):', score)

        val_score = self.model.evaluate(X_val, y_val)
        print('Validation accuracy (all):', val_score)
        self.val_score = '{:.6f}'.format(val_score[1])

    def predict(self, X):
        X = self.process_bow(X)
        Y = self.model.predict(X, verbose=1)
        return Y

    def fit_tokenizer(self, train_text, train_nolabel_text, test_text):
        filters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\'\t\n'
        print(" ======= FIT tokenizer ======= ")
        self.tokenizer = Tokenizer(filters=filters, split=" ", num_words=self.tokenNumWords)
        self.tokenizer.fit_on_texts(train_text + train_nolabel_text + test_text)
        print(" ======= Complete ======= ")

    def process_bow(self, text):
        bow = self.tokenizer.texts_to_matrix(text, mode='count')
        return bow

    def saveTokenizer(self, token_name=config.TOKENIZER_NAME):
        cPickle.dump(self.tokenizer, open("./metadata/" + token_name + ".pickle", 'wb'))

    def loadTokenizer(self, token_name=config.TOKENIZER_NAME):
        self.tokenizer = cPickle.load(open("./metadata/" + token_name + ".pickle", 'rb'))

    def saveModel(self, model_name=config.VERSION_NAME):
        self.model.save(os.path.join('./model', model_name + '.h5'))

    def loadModel(self, model_name=config.VERSION_NAME):
        from keras.models import load_model
        self.model = load_model(os.path.join('./model', model_name + '.h5'))

    def split_validation(self, X_all, Y_all, percentage):
        from math import log, floor
        def _shuffle(X, Y):
            np.random.seed(1)
            randomize = np.arange(len(X))
            np.random.shuffle(randomize)
            return (X[randomize], Y[randomize])

        all_data_size = len(X_all)
        valid_data_size = int(floor(all_data_size * percentage))

        X_all, Y_all = _shuffle(X_all, Y_all)

        X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
        X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

        return X_train, Y_train, X_valid, Y_valid

    def saveHistory(self, name=config.VERSION_NAME, dir_path='./history'):
        # print(self.history.history['acc'])

        if len(name) == 0:
            fw = open(os.path.join(dir_path, self.val_score + '.pickle'), 'wb')
        else:
            fw = open(os.path.join(dir_path, name + '.pickle'), 'wb')
        cPickle.dump(self.history.history, fw)



class DataGenerator(object):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def generate(self, X, y):
        while 1:
            # X, y = self._shuffle(X, y)

            imax = int(len(y) / self.batch_size)
            for i in range(imax):
                yield X[i*self.batch_size:(i+1)*self.batch_size], y[i*self.batch_size:(i+1)*self.batch_size]

    # def _shuffle(self, X, Y):
    #     np.random.seed(1)
    #     randomize = np.arange(len(X))
    #     np.random.shuffle(randomize)
    #     return (X[randomize], Y[randomize])