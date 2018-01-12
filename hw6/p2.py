#coding:utf-8
from gensim.models.word2vec import Word2Vec

import adjustText
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False


import numpy as np
import jieba
jieba.set_dictionary('jieba/dict.txt.big')

def csvReader(path='./data/all_sents.txt'):
    ret = []
    word_set = set()

    for line in open(path, 'r').readlines():
        seg_list = jieba.cut(line.replace("\n","").replace("\"","").replace("，","").replace(".","")
                             .replace("「", "").replace("」", "").replace("、", "").replace("0", "")
                             .replace("1", "")
                             .replace("2", "")
                             .replace("3", "")
                             .replace("4", "")
                             .replace("5", "")
                             .replace("6", "")
                             .replace("7", "")
                             .replace("8", "")
                             .replace("9", "")
                              )
        word_set.add(x for x in seg_list)
        # print([x for x in seg_list])
        ret.append([x for x in seg_list])

    print(len(word_set))

    return ret


def word2vec(sentences):

    model = Word2Vec(sentences, size=300, window=5)
    model.save('./model/p2.w2v')


def getWordVector():

    words = []
    word_vectors = []

    model = Word2Vec.load('./model/p2.w2v')
    for word, vocab_obj in model.wv.vocab.items():
        if( vocab_obj.count > 5000 and len(word) > 0):
            # print(word, vocab_obj.count)
            words.append(word)
            word_vectors.append(model[word])
    return np.array(word_vectors), words


def plot(word_vector, words):
    print("== start SVD ==")
    mean = np.mean(word_vector, axis=0)
    vector_mean = word_vector - mean
    U, S, V = np.linalg.svd(vector_mean.transpose(), full_matrices=False)
    vis_data = np.dot(word_vector, U[:, :2])
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    print("== plt figure ==")
    plt.figure()
    plt.plot(vis_x, vis_y, 'o')
    texts = [plt.text(X, Y, Text) for X, Y, Text in zip(vis_x, vis_y, words)]
    plt.title(str(adjustText.adjust_text(texts, vis_x, vis_y, arrowprops=dict(arrowstyle="->", color='red'))))
    plt.savefig("Q2_2.png")


if __name__ == "__main__":

    # print("read data")
    # sentences = csvReader()
    #
    # print("word2vec")
    # word2vec(sentences)

    print("wordVector")
    word_vectors, words = getWordVector()
    print("words len: ", len(words))

    print("plot")
    plot(word_vectors, words)

