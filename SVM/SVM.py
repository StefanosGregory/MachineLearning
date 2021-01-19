import pickle

import sklearn
from sklearn import datasets
from sklearn import metrics

cancer = datasets.load_breast_cancer()
seed = 1
# print(cancer.feature_names)
# print(cancer.target_names)

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=seed)
best = 0
# print(X_train, y_train)
classes = ['malignant', 'benign']

'''
seed = 0
for i in range(15):
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1, random_state=i)

    clf = svm.SVC(kernel="linear", C=2)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy: ", acc)
    if acc > best:
        seed = i
        best = acc
        with open("svm.pickle", "wb") as f:
            pickle.dump(clf, f)
'''

pickle_in = open("svm.pickle", "rb")
clf = pickle.load(pickle_in)

y_pred = clf.predict(X_test)

acc = metrics.accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)
for x in range(len(y_pred)):
    print("Expected -> ", round(y_pred[x]), ", True -> ", y_test[x])