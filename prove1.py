from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from kneighbors import KNeighbors
import csv

# possible types mapped to numbers
irisValues = {}

# "predicts" setosa every time
class HardCodedModel:
    def predict(self, data):
        result = []
        for datum in data:
            result.append(0)
        return result


class HardCodedClassifier:
    def fit(self, data, target):
        return HardCodedModel()

path = input("Path to CSV file (keep blank for default iris data): ")
if (path == ""):
    # default to using the datasets
    iris = datasets.load_digits()
    data = iris.data
    target = iris.target
else:
    # read the csv file and populate the data and target
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
                if row[4] not in irisValues:
                    irisValues[row[4]] = len(irisValues)
                target.append(irisValues[row[4]])


#split into training and test sets
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.3, train_size=.7, shuffle=True)

classifiers = [
    [0, "Gaussian NB", GaussianNB],
    [1, "Hard Coded", HardCodedClassifier],
    [2, "K-Nearest Neighbors", KNeighbors],
    [3, "Built-in K-Nearerst Neighbors", KNeighborsClassifier]
]

print("Choose a classifier:")
for key, name, classifier in classifiers:
    print("\t{} : {}".format(key, name))
classKey = int(input(">"))



# Use Gaussian NB to predict
classifier = classifiers[classKey][2]()
model = classifier.fit(data_train, target_train)
targets_predicted = model.predict(data_test)
score = accuracy_score(target_test, targets_predicted)
print(classifiers[classKey][1] + " score: " + ("%.1f" % (score * 100)) + "%")