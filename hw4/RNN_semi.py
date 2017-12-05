
import _pickle as cPickle
import os
import numpy as np
import config
import sys

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
from keras.utils import to_categorical


class Model:

    def __init__(self, seqlen = 35, STEPS_PER_EPOCH=30,):
        self.tokenizer = None
        self.tokenNumWords = 10000
        self.val_score = None
        self.seqlen = seqlen
        self.w2v_features = 80

        self.BATCH = 128
        self.EPOCH = 300
        self.STEPS_PER_EPOCH = STEPS_PER_EPOCH

        self.train_acc = []
        self.val_acc = []

    def init(self):
        self.model = Sequential()
        print("Max len of tokenizer.word", self.tokenNumWords)

        # self.model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
        # self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        # self.model.add(MaxPooling1D(pool_size=2))
        # self.model.add(LSTM(100))
        # self.model.add(Dense(1, activation='sigmoid'))
        #
        # self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        # self.model.summary()


        # self.model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
        # self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Bidirectional(LSTM(128,
                                          activation='tanh', dropout=0.1, return_sequences=True),
                                     input_shape=(self.seqlen, self.w2v_features)))
        self.model.add(Bidirectional(LSTM(64, activation='tanh', dropout=0.1)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2, activation='softmax'))

        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    def train(self, X, y, Xraw):

        X_train, y_train, X_val, y_val = self.split_validation(X, y, 0.9)



        # Start Semi Learning

        for round in range(10):
            print("Train data, Round ", str(round))

            train_generator = DataGenerator(self.BATCH).generate(X_train, y_train)
            filepath = os.path.join('./model', config.VERSION_NAME + "_Round" + str(round)
                                    + "_{epoch:02d}_{val_acc:.6f}.h5")
            model_checkpoint = ModelCheckpoint(filepath, verbose=1, period=1, save_best_only=True, monitor='val_acc')
            history = self.model.fit_generator(train_generator, steps_per_epoch=self.STEPS_PER_EPOCH,
                                               epochs=int(len(y_train)/self.STEPS_PER_EPOCH/self.BATCH),
                                               verbose=1,validation_data=(X_val, y_val), callbacks=[model_checkpoint])
            self.train_acc += history.history['acc']
            self.val_acc += history.history['val_acc']

            for part in range(int(len(Xraw)/250000)):
                print("Predict Semi Data, Round ", str(round), " Part:", part)
                Xno = self.process_w2v(Xraw[250000*part: 250000*(part+1)])

                Yno = self.model.predict(Xno, verbose=1)

                X_semi_no, y_semi_no = self.filter_semiData(Xno, Yno)
                y_semi_no = np.argmax(y_semi_no, axis=1) # Hard Semi
                y_semi_no = to_categorical(y_semi_no,2)

                # print(y_train[0:5])
                # print(y_semi_no[0:5])

                print("Len of semi:", len(y_semi_no))

                filepath = os.path.join('./model', config.VERSION_NAME+"_Round" + str(round)+"_part" + str(part)
                                        + "_{epoch:02d}_{val_acc:.6f}.h5")
                model_checkpoint = ModelCheckpoint(filepath, verbose=1, period=1, save_best_only=True, monitor='val_acc')
                train_generator = DataGenerator(self.BATCH).generate(X_semi_no, y_semi_no)
                history = self.model.fit_generator(train_generator, steps_per_epoch=self.STEPS_PER_EPOCH,
                                                   epochs=int(len(y_semi_no)/self.STEPS_PER_EPOCH/self.BATCH)+1,
                                                   verbose=1, validation_data=(X_val, y_val), callbacks=[model_checkpoint])
                self.train_acc += history.history['acc']
                self.val_acc += history.history['val_acc']

                self.saveHistory()

        score = self.model.evaluate(X, y)
        print('Train accuracy (all):', score)

        val_score = self.model.evaluate(X_val, y_val)
        print('Validation accuracy (all):', val_score)
        self.val_score = '{:.6f}'.format(val_score[1])

    def filter_semiData(self, Xno, Yno):
        print("filt Semi Data...")
        threshhold = 0.25
        index = [(y[0] < threshhold and y[1] > 1-threshhold) or (y[0] > 1-threshhold and y[1] < threshhold)
                 for y in Yno]
        return Xno[index], Yno[index]

    def predict(self, X):
        Y = self.model.predict(X, verbose=1)
        # return np.array([1 if p > 0.5 else 0 for p in Y])
        return Y

    def fit_tokenizer(self, train_text, train_nolabel_text, test_text):
        filters = '"#$%&()*+-/:;<=>@[\\]^_`{|}~\'\t\n'
        print(" ======= FIT tokenizer ======= ")
        self.tokenizer = Tokenizer(filters=filters, split=" ", num_words=self.tokenNumWords)
        self.tokenizer.fit_on_texts(train_text + train_nolabel_text + test_text)
        print(" ======= Complete ======= ")

    def process_text(self, text):
        sequence = self.tokenizer.texts_to_sequences(text)
        return pad_sequences(sequence, maxlen=self.seqlen)

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
        cPickle.dump((self.train_acc, self.val_acc), fw)


    def fit_w2v(self, train_text, train_nolabel_text, test_text):
        if not os.path.isfile(os.path.join('./model/', config.WORD2VEC_NAME + '.w2v')):
            print("== Start to fit w2v")

            text_list = train_text + train_nolabel_text + test_text
            print(text_list[0:5])

            sentences = []
            for text in text_list:
                sentences.append(text.split(" "))
            print(sentences[0:5])

            model = Word2Vec(sentences, size=self.w2v_features, window=5)
            model.save(os.path.join('./model/', config.WORD2VEC_NAME + '.w2v'))
            print("== Complete fitting w2v")
            return
        print("== Already fit w2v")

    def process_w2v(self, text_list):
        matrix = np.zeros(shape=(len(text_list), self.seqlen, self.w2v_features))

        model = Word2Vec.load(os.path.join('./model/', config.WORD2VEC_NAME + '.w2v'))
        print(model)
        for i, sentence in enumerate(text_list):
            words = sentence.split(" ")
            for j in range(min(len(words), self.seqlen)):
                if words[j] not in model:
                    print(words[j], ":::", words)
                    continue
                matrix[i, j, :] = model[words[j]]

            sys.stdout.write("\rprocess_w2v: " + str(i) + " / " + str(len(text_list)))
            sys.stdout.flush()

        print("== peak process w2v:")
        print(matrix[5,:,:])
        print("==")
        return matrix




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