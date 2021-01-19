import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

seed = 3

# Read data from csv file
data = pd.read_csv("student-mat.csv", sep=";")
# Take only care about columns
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
# print(data.head())
# Predict column
predict = "G3"
# Split into 2 columns X = train data, y = predict data
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
'''
best = 0
for i in range(100):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
    # Create model
    linear = linear_model.LinearRegression()
    # Train model
    linear.fit(X_train, y_train)
    acc = linear.score(X_test, y_test)
    print("Accuracy: ", acc)
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f) '''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
print("Co: ",  linear.coef_)
print("Intercept: ", linear.intercept_)

predictions = linear.predict(X_test)

for x in range(len(predictions)):
    print("Expected -> ", round(predictions[x]), "True -> ", y_test[x])
p = 'G1'
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()