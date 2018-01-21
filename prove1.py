from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from kneighbors import KNeighbors
from hardcoded import HardCodedClassifier
import csv

# possible types mapped to numbers
irisValues = {}


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