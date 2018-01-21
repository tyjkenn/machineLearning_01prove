import numpy
import statistics
from collections import Counter

def sqrdist(a, b):
    distance = 0;
    for i in range(len(a)):
        distance += (a[i] - b[i])**2
    return distance

def findMode(options):
    c = Counter(options)
    return c.most_common(1)[0][0]

class KNeighborsModel:
    def __init__(self, data, target, k=None):
        if k is None:
            k = int(input("Choose a number for K:"))
        self.targets = target
        self.k = k
        self.stds = []
        self.means = []
        cols = zip(*data)
        for col in cols:
            self.stds.append(numpy.std(col))
            self.means.append(numpy.mean(col))
        self.data = self.normalize(data)

    def normalize(self, data):
        for i in range(len(data)):
            data[i] = self.normalizeRow(data[i])
        return data

    def normalizeRow(self, row):
        for i in range(len(row)):
            x = (row[i] - self.means[i])
            if self.stds[i] != 0:
                x /= self.stds[i]
            row[i] = x
        return row


    def predict(self, data):
        results = []
        data = self.normalize(data)
        for row in data:
            records = []
            #get the k closest
            for i in range(len(self.data)):
                dist = sqrdist(self.data[i], row)
                if len(records) == 0:
                    records.append([self.targets[i], dist])
                else:
                    inserted = False;
                    for j in range(len(records)):
                        if dist < records[j][1]:
                            records.insert(j, [self.targets[i], dist])
                            inserted = True
                            if len(records) > self.k:
                                records.pop()
                            break;

                    if not inserted and len(records) < self.k:
                        records.append([self.targets[i], dist])
            #find the mode
            ktargets = next(zip(*records))
            results.append(findMode(ktargets))
        return results


class KNeighbors:
    def fit(self, data, target, k=None):
        return KNeighborsModel(data, target, k)