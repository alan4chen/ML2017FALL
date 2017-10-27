import csv
import numpy as np

ITEM_LIST = "age","fnlwgt","sex","capital_gain","capital_loss","hours_per_week","Federal-gov","Local-gov","Never-worked","Private","Self-emp-inc","Self-emp-not-inc","State-gov","Without-pay","?_workclass","10th","11th","12th","1st-4th","5th-6th","7th-8th","9th","Assoc-acdm","Assoc-voc","Bachelors","Doctorate","HS-grad","Masters","Preschool","Prof-school","Some-college","Divorced","Married-AF-spouse","Married-civ-spouse","Married-spouse-absent","Never-married","Separated","Widowed","Adm-clerical","Armed-Forces","Craft-repair","Exec-managerial","Farming-fishing","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-serv","Prof-specialty","Protective-serv","Sales","Tech-support","Transport-moving","?_occupation","Husband","Not-in-family","Other-relative","Own-child","Unmarried","Wife","Amer-Indian-Eskimo","Asian-Pac-Islander","Black","Other","White","Cambodia","Canada","China","Columbia","Cuba","Dominican-Republic","Ecuador","El-Salvador","England","France","Germany","Greece","Guatemala","Haiti","Holand-Netherlands","Honduras","Hong","Hungary","India","Iran","Ireland","Italy","Jamaica","Japan","Laos","Mexico","Nicaragua","Outlying-US(Guam-USVI-etc)","Peru","Philippines","Poland","Portugal","Puerto-Rico","Scotland","South","Taiwan","Thailand","Trinadad&Tobago","United-States","Vietnam","Yugoslavia","?_native_country"
Item2ID = dict()
for item_id, item in enumerate(ITEM_LIST):
    Item2ID[item] = item_id


def readXdata(path):
    tmp = []
    # Read Data
    with open(path, 'r') as f:
        rd = csv.reader(f, delimiter=',')
        for line in rd:
            if line[0] == "age":
                continue
            else:
                tmp.append(np.array([float(_) for _ in line]))
    X = np.array(tmp)
    return X

if __name__ == "__main__":

    X_train = readXdata("X_train")
    X_test = readXdata("X_test")

    np.save("./X_train.npy", X_train)
    np.save("./X_test.npy", X_test)

