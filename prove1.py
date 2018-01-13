from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import csv

irisValues = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

path = input("Path to CSV file: ")
if (path == ""):
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
else:
    with open(path) as file:
        csvRead = csv.reader(file, delimiter=',')
        data = []
        target = []
        for row in csvRead:
            if (len(row) == 5):
                rowVals = []
                rowVals.append(float(row[0]))
                rowVals.append(float(row[1]))
                rowVals.append(float(row[2]))
                rowVals.append(float(row[3]))
                data.append(rowVals)
                target.append(irisValues[row[4]])



data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.3, train_size=.7, shuffle=True)

# Use Gaussian NB to predict
classifier = GaussianNB()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)
score = accuracy_score(target_test, targets_predicted)
print("GaussianNB score: " + ("%.1f" % (score * 100)) + "%")


class HardCodedModel:
    def predict(self, data):
        result = []
        for datum in data:
            result.append(0)
        return result


class HardCodedClassifier:
    def fit(self, data, target):
        return HardCodedModel()


# Use my terrible classifier
classifier = HardCodedClassifier()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)
score = accuracy_score(target_test, targets_predicted)
print("HardCodedClassifier score: " + ("%.1f" % (score * 100)) + "%")