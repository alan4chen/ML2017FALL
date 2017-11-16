import sys
import itertools
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix

from config import VERSION_NAME
import csv_converter
from cnn import Model

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{:.3f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="red" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def main(args):
    class_names = ['Angry', 'Digust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    width = height = 48

    X, y = csv_converter.readTrainNpy()
    model = Model()
    model.loadModel(name=VERSION_NAME)

    X_train, y_train, X_val, y_val = model.split_validation(X, y, 0.9)

    Y_val = model.predict(X_val)
    cnf_matrix = confusion_matrix(np.argmax(Y_val, axis=1), np.argmax(y_val, axis=1))

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
            title='Confusion Matrix')

    from PIL import Image
    plt.savefig('Q3.png')
    Image.open('Q3.png').save('Q3.png', 'PNG')

if __name__ == '__main__':
    main(sys.argv)