from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from kneighbors import KNeighbors
from hardcoded import HardCodedClassifier
import numpy as np
import csv
import preprocessor

# possible types mapped to numbers
irisValues = {}

preprocessors = [
    [0, "Car Evaluation", preprocessor.car_eval],
    [1, "Diabetes", preprocessor.diabetes]
]

classifiers = [
    [0, "Gaussian NB", GaussianNB],
    [1, "Hard Coded", HardCodedClassifier],
    [2, "K-Nearest Neighbors", KNeighbors],
    [3, "Built-in K-Nearerst Neighbors", KNeighborsClassifier]
]


useSklearn = input("Will you use one of sklearn's data sets?(y/n)")
if useSklearn == "y" or useSklearn == "Y":
    print("Choose one of the following:")
    print("0 : Iris")
    print("1 : Wine")
    print("2 : Digits")
    option = int(input(">"))
    if option == 0:
        dataset = datasets.load_iris()
    elif option == 1:
        dataset = datasets.load_wine()
    else:
        dataset = datasets.load_digits()
    data = dataset.data
    target = dataset.target
else:
    path = input("Path to CSV file: ")
    print("Choose a preprocessor:")
    for key, name, preproc in preprocessors:
        print("\t{} : {}".format(key, name))
    preprocKey = int(input(">"))
    data, target = preprocessors[preprocKey][2](path)


# split into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.3, train_size=.7, shuffle=True)

print("Choose a classifier:")
for key, name, classifier in classifiers:
    print("\t{} : {}".format(key, name))
classKey = int(input(">"))


classifier = classifiers[classKey][2]()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)
score = accuracy_score(target_test, targets_predicted)
print(classifiers[classKey][1] + " score: " + ("%.1f" % (score * 100)) + "%")