
import os
import numpy as np
import keras.backend as K
from csv_reader import readMovies

dict_id2genre = {0: "Animation",
1: "Adventure",
2: "Comedy",
3: "Action",
4: "Drama",
5: "Thriller",
6: "Crime",
7: "Romance",
8: "Children's",
9: "Documentary",
10: "Sci-Fi",
11: "Horror",
12: "Western",
13: "Mystery",
14: "Film-Noir",
15: "War",
16: "Fantasy",
17: "Musical"}

dict_id2label = {0: 2,
1: 2,
2: 2,
3: 0,
4: 0,
5: 1,
6: 1,
7: 0,
8: 2,
9: 0,
10: 0,
11: 1,
12: 1,
13: 0,
14: 1,
15: 1,
16: 2,
17: 2}

def rmse(y_true, y_pred):
    return K.sqrt(K.mean((y_pred - y_true) ** 2))

def draw_single(x, y):
    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    y = np.array(y)
    x = np.array(x, dtype=np.float64)

    vis_data = TSNE(n_components=2).fit_transform(x)

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cm = plt.cm.get_cmap("RdYlBu")

    for i in range(18):
        print("plot:", i)
        plt.figure()
        plt.scatter(vis_x, vis_y, c=[0 if yb==i else 1 for yb in y], cmap=cm, s=2)
        plt.title(str(i) + " genre:" + dict_id2genre[i])

        plt.savefig("Q5_"+str(i)+'.png')


def draw(x, y):
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    from matplotlib import pyplot as plt
    from sklearn.manifold import TSNE
    y = np.array(y)
    x = np.array(x, dtype=np.float64)

    vis_data = TSNE(n_components=2).fit_transform(x)

    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]

    cm = plt.cm.get_cmap("RdYlGn")
    plt.figure()
    plt.scatter(vis_x, vis_y, c=[dict_id2label[yb] for yb in y], cmap=cm, s=2)

    plt.savefig("Q5_lcy_all.png")


if __name__ == "__main__":
    from keras.models import load_model
    model = load_model(os.path.join('./model', 'prev_model.h5'), custom_objects={'rmse': rmse})

    movieID_np, movieCAT_np = readMovies(readCategory=True)




    ret = np.array(model.layers[3].get_weights()).squeeze()
    print(ret.shape)

    draw(ret[list(movieID_np)], movieCAT_np)
