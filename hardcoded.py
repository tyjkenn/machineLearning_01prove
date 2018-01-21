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