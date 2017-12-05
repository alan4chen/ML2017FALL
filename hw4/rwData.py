import sys
import os
import numpy as np
import _pickle as cPickle

from stemmer import LancasterStemmer
from stopper import isStopWord, duplicateRemover, CHAR_TO_REMOVE
st = LancasterStemmer()

import config

# 1 +++$+++ are wtf ... awww thanks !

def readTrainingLabelTXT(training_data_path = './data/training_label.txt'):
    from keras.utils import to_categorical
    fo = open(training_data_path, 'r')
    label = []
    text = []
    for line in fo.readlines():
        splitted = line.split(" ")
        label.append(float(splitted[0]))
        text.append(line[10:])
    return to_categorical(label,2), text
    # return np.array(label), text

def readTrainingNoLabelTXT(training_nolabel_path = './data/training_nolabel.txt'):
    fo = open(training_nolabel_path, 'r')
    text = []
    for line in fo.readlines():
        text.append(line)
    return text

def readTestingTXT(testing_path = './data/testing_data.txt'):
    fo = open(testing_path, 'r')
    text = []
    for idx, line in enumerate(fo.readlines()):
        if idx == 0: continue
        l = len(str(idx-1)+",")
        text.append(line[l:])
    return text

def readTrainingLabelTXTstemed(training_data_path = './data/training_label.txt'):
    # if not os.path.isfile('./data/training_label.pickle'):
        training_label, training_text = readTrainingLabelTXT()
        training_text_stemmed = []
        for sentence in training_text:
            sentence_stemmed = ""
            sentence = sentence.translate({ord(i): None for i in CHAR_TO_REMOVE})
            for word in sentence.split(" "):
                if isStopWord(word):
                    continue
                if word.isdigit():
                    word = 'NUMBER'
                if len(word) < 1: continue
                sentence_stemmed += duplicateRemover(st.stem(word))
                sentence_stemmed += " "
            training_text_stemmed.append(sentence_stemmed[:-1])
        cPickle.dump((training_label, training_text_stemmed), open('./data/training_label.pickle', 'wb'))
        return training_label, training_text_stemmed
    # obj = cPickle.load(open('./data/training_label.pickle', 'rb'))
    # print("Loaded: TrainingLabelTXT")
    # return obj[0], obj[1]



def readTrainingNoLabelTXTstemed(training_nolabel_path = './data/training_nolabel.txt'):
    # if not os.path.isfile('./data/training_nolabel.pickle'):
        training_nolabel_text = readTrainingNoLabelTXT()
        training_nolabel_stemmed = []
        for sentence in training_nolabel_text:
            sentence = sentence.translate({ord(i): None for i in CHAR_TO_REMOVE})
            sentence_stemmed = ""
            for word in sentence.split(" "):
                if isStopWord(word): continue
                if word.isdigit():
                    word = 'NUMBER'
                if len(word) < 1: continue
                sentence_stemmed += duplicateRemover(st.stem(word))
                sentence_stemmed += " "
            training_nolabel_stemmed.append(sentence_stemmed[:-1])
        cPickle.dump(training_nolabel_stemmed, open('./data/training_nolabel.pickle', 'wb'))
        return training_nolabel_stemmed
    # print("Loaded: TrainingNoLabelTXT")
    # return cPickle.load(open('./data/training_nolabel.pickle', 'rb'))

def readQ3(path):
    training_nolabel_text = readTrainingNoLabelTXT(path)
    training_nolabel_stemmed = []
    for sentence in training_nolabel_text:
        sentence = sentence.translate({ord(i): None for i in CHAR_TO_REMOVE})
        sentence_stemmed = ""
        for word in sentence.split(" "):
            if isStopWord(word): continue
            if word.isdigit():
                word = 'NUMBER'
            if len(word) < 1: continue
            sentence_stemmed += duplicateRemover(st.stem(word))
            sentence_stemmed += " "
        training_nolabel_stemmed.append(sentence_stemmed[:-1])
    cPickle.dump(training_nolabel_stemmed, open('./data/training_nolabel.pickle', 'wb'))
    return training_nolabel_stemmed


def readTestingTXTstemed(testing_path = './data/testing_data.txt'):
    # if not os.path.isfile('./data/testing_data.pickle'):
        testing_text = readTestingTXT()
        testing_text_stemmed = []
        for sentence in testing_text:
            sentence = sentence.translate({ord(i): None for i in CHAR_TO_REMOVE})
            sentence_stemmed = ""
            for word in sentence.split(" "):
                if isStopWord(word): continue
                if word.isdigit():
                    word = 'NUMBER'
                if len(word) < 1: continue
                sentence_stemmed += duplicateRemover(st.stem(word))
                sentence_stemmed += " "
            testing_text_stemmed.append(sentence_stemmed[:-1])
        cPickle.dump(testing_text_stemmed, open('./data/testing_data.pickle', 'wb'))
        return testing_text_stemmed
    # print("Loaded: TestingTxt")
    # return cPickle.load(open('./data/testing_data.pickle', 'rb'))

def writeAnsPath(result, filepath):
    with open(filepath, 'w') as f:
        f.write('id,label\n')
        for i in range(len(result)):
            predict = np.argmax(result[i])
            # predict = result[i]
            f.write(repr(i) + ',' + repr(predict) + '\n')




if __name__ == "__main__":
    training_label, training_text = readTrainingLabelTXTstemed()
    print(training_label[0:3], training_text[0:3])
    training_nolabel_text = readTrainingNoLabelTXTstemed()
    print(training_nolabel_text[0:3])
    testing_text = readTestingTXTstemed()
    print(testing_text[0:3])

