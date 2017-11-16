import os

import numpy as np
import csv_converter
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from config import MODEL_SAVE_DIR, VERSION_NAME


"""

cnn 111312

435 epochs  val:0.678161,  public: 0.67874  cnn111312_434_0.678161.h5


cnn 111201
400 epochs  public: 0.67093
500 epcohs  public: 0.66313

"""

class Model():
    def __init__(self):
        self.VAL = 2400
        self.BATCH = 128
        self.EPOCHS = 600
        self.score = ''
        self.history = None

    def init(self):
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='valid'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.05))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Dropout(0.15))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dropout(0.3))
        self.model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(MaxPooling2D((2, 2)))

        self.model.add(Flatten())
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=256, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=64, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=32, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=7, activation='softmax'))
        self.model.summary()

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X, y):

        # Split Out Validation Data
        X = X / 225
        X = X.reshape(X.shape[0], 48, 48, 1)

        X_train, y_train, X_val, y_val = self.split_validation(X, y, 0.9)

        # Use Original Train
        self.history =self.model.fit(X_train, y_train, batch_size=self.BATCH, epochs=100, verbose=1, 
            validation_data=(X_val, y_val))
        self.saveHistory(name=VERSION_NAME+"_ORI")

        # Use Generator
        gen = ImageDataGenerator(rotation_range=30, horizontal_flip=True,
                                 zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1,)

        filepath = os.path.join(MODEL_SAVE_DIR, VERSION_NAME+"_{epoch:02d}_{val_acc:.6f}.h5")
        model_checkpoint = ModelCheckpoint(filepath, verbose=1, period=1)

        self.history = self.model.fit_generator(gen.flow(X_train, y_train, batch_size=self.BATCH, seed=1), steps_per_epoch=int(X.shape[0]/self.BATCH)+1,
                                 epochs= self.EPOCHS, validation_data=(X_val, y_val), callbacks=[model_checkpoint])

        score = self.model.evaluate(X, y)
        self.score = '{:.6f}'.format(score[1])
        print('Train accuracy (all):', self.score)

    def predict(self, X):
        X = X / 225
        X = X.reshape(X.shape[0], 48, 48, 1)
        return self.model.predict(X)

    def saveModel(self, name='', dir_path='./model'):
        if len(name) == 0:
            self.model.save(os.path.join(dir_path, self.score + '.h5'))
        else:
            self.model.save(os.path.join(dir_path, name + '.h5'))

    def loadModel(self, name='', dir_path='./model'):
        from keras.models import load_model
        self.model = load_model(os.path.join(dir_path, name + '.h5'))

    def saveHistory(self, name='', dir_path='./history'):
        import _pickle as cPickle

        print(self.history.history['acc'])

        if len(name) == 0:
            fw = open(os.path.join(dir_path, self.score + '.pickle'), 'wb')
        else:
            fw = open(os.path.join(dir_path, name + '.pickle'), 'wb')
        cPickle.dump(self.history.history, fw)


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

