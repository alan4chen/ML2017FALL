import sys

index_to_word = dict()
word_to_count = dict()
index = 0

with open(sys.argv[1], "r") as f:
    text = f.read()
    text = text.replace("\n", "").replace("\r","")
    t_list = text.split(" ")

    for word in t_list:
        if word not in word_to_count:
            index_to_word[index] = word
            index += 1
            word_to_count[word] = 0

        word_to_count[word] += 1


for i in range(index):
    word = index_to_word[i]
    count = word_to_count[word]
    print(str(word) + " " + str(i) + " " + str(count))
