import itertools

CHAR_TO_REMOVE = '"#$%&()*+-/:;<=>@[\\]^_`{|}\'\t\n'

lst = ['is', 'are', 'a', 'the', 'm', 'was', 'were']
stopWordsSet = set()
for t in lst:
    stopWordsSet.add(t)

def isStopWord(text):
    return text in stopWordsSet


def duplicateRemover(text):
    #     return ''.join(ch for ch, _ in itertools.groupby(text))

    lst = list(x[:2] for x in list(list(cc) for ch, cc in itertools.groupby(text)))
    return ''.join(list(''.join(item) for item in lst))


if __name__ == '__main__':
    print(isStopWord('are'))
    print(isStopWord('are2'))

    print(duplicateRemover('biggg'))
    print(duplicateRemover('goooood'))
