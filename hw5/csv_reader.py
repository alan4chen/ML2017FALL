import numpy as np

def to_categorical(index, categories):
    categorical = np.zeros(categories, dtype=int)
    categorical[index] = 1
    return list(categorical)

def readUsers(path="./data/users.csv"):

    users_data = []

    f = open(path, 'r')
    for line in f.readlines()[1:]:
        if len(line) != 0:
            userID, gender, age, occu, zipcode = line[:-1].split('::')
            users_data.append((int(userID), 0 if gender is 'M' else 1, int(age), int(occu)))
    users_data.sort(key=lambda x: x[0])

    userID_np = np.array([int(x[0]) for x in users_data])
    gender_np = np.array([int(x[1]) for x in users_data])
    age_np = np.array([int(x[2]) for x in users_data])
    occu_np = np.array([x[3] for x in users_data]) # TODO: enbedding occu
    return userID_np, gender_np, age_np, occu_np

def readMovies(path = "./data/movies.csv", readCategory = False):
    f = open(path, 'r', encoding='latin-1')

    movies_data = []

    for line in f.readlines()[1:]:
        if len(line) != 0:
            movieID, title, genre = line[:-1].split('::')
            movies_data.append((movieID, title, genre))


    movieID_np = np.array([int(x[0]) for x in movies_data])

    if readCategory:
        import random
        random.seed(0)
        movieCats = [random.choice(x[2].split("|")) for x in movies_data]
        cat_to_label = {}
        counter = 0
        movieCAT_list = []
        for cat in movieCats:
            if cat not in cat_to_label:
                cat_to_label[cat] = counter
                counter += 1
                print(cat, " : ", counter)
            movieCAT_list.append(cat_to_label[cat])
        print("total categories: ", counter)
        return movieID_np, np.array(movieCAT_list)

    else:
        return movieID_np

def readTrain(path = "./data/train.csv"):
    f = open(path, 'r')

    train_data = []
    for line in f.readlines()[1:]:
        if len(line) != 0:
            dataID, userID, movieID, ratings = line[:-1].split(",")
            train_data.append((userID, movieID, ratings))

    userID_np = np.array([int(x[0]) for x in train_data])
    movieID_np = np.array([int(x[1]) for x in train_data])
    rating_np = np.array([float(x[2]) for x in train_data])

    return userID_np, movieID_np, rating_np

def readTest(path = "./data/test.csv"):
    f = open(path, 'r')

    test_data = []
    for line in f.readlines()[1:]:
        if len(line) != 0:
            TestDataID, UserID, MovieID = line[:-1].split(",")
            test_data.append((UserID, MovieID))

    userID_np = np.array([int(x[0]) for x in test_data])
    movieID_np = np.array([int(x[1]) for x in test_data])

    return userID_np, movieID_np


if __name__ == "__main__":
    a, b, c, d = readUsers()
    print(a)


def writeAns(ans, filepath):
    with open(filepath, 'w') as f:
        f.write('TestDataID,Rating\n')
        for i in range(len(ans)):
            f.write(repr(i+1) + ',' + repr(float(ans[i])) + '\n')