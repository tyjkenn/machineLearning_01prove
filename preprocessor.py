import pandas as pd
import numpy as np

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
    data = data.drop("class", axis=1)
    return data.values, target.values


def diabetes(filename):
    data = pd.read_csv(filename, dtype=None,
                       names=["timesPreg", "plasmaGlucose", "bloodPress", "skinThick", "serum", "bmi", "pedigree", "age", "class"],
                       sep=',', skipinitialspace=True, )
    data[["plasmaGlucose", "bloodPress", "skinThick", "serum", "bmi", "pedigree", "age"]] = data[["plasmaGlucose", "bloodPress", "skinThick", "serum", "bmi", "pedigree", "age"]].replace(0, np.NaN)
    data.fillna(data.mean(), inplace=True)
    target = data["class"]
    data = data.drop("class", axis=1)
    return data.values, target.values


def mpg(filename):
    data = pd.read_csv(filename, dtype={'horsepower': np.float64}, na_values=["?"],
                       names=["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year",
                              "origin", "name"],
                       delim_whitespace=True)
    data['horsepower'] = data['horsepower'].replace('?', np.nan)
    data['horsepower'].fillna((data['horsepower'].mean()), inplace=True)
    target = data["mpg"]
    data = data.drop("mpg", axis=1)
    data = data.drop("name", axis=1)
    return data.values, target.values
