import pandas as pd

def order_strings(column, order):
    column[:] = [order.index(x) for x in column]
    return column

def car_eval(filename):
    data = pd.read_csv(filename, dtype=None,
                         names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"],
                         sep=',', skipinitialspace=True)
    data["buying"] = data["buying"].astype('category').cat.codes
    data["maint"] = data["maint"].astype('category').cat.codes
    data["doors"] = data["doors"].astype('category').cat.codes
    data["persons"] = data["persons"].astype('category').cat.codes
    data["lug_boot"] = data["lug_boot"].astype('category').cat.codes
    data["safety"] = data["safety"].astype('category').cat.codes
    data["class"] = data["class"].astype('category').cat.codes
    target = data["class"]
    data.drop("class", axis=1)
    return data.values, target.values

def diabetes(filename):
    pass

def mpg(csv):
    pass