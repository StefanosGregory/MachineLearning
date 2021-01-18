import pickle

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
# print(data.head())

le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
best = 0
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

'''
for i in range(100):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=5)

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print("Accuracy: ", acc)
    if acc > best:
            best = acc
            with open("car.pickle", "wb") as f:
                pickle.dump(model, f)
'''

pickle_in = open("car.pickle", "rb")
model = pickle.load(pickle_in)
acc = model.score(X_test, y_test)
print("Accuracy: ", acc)
predicted = model.predict(X_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Actual: ", names[y_test[x]])
    n = model.kneighbors([X_test[x]], 9, True)
    print("N: ", n)