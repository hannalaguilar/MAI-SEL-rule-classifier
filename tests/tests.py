from src.limpio import DataCSV, Selector, Rule
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import numpy as np


# data = DataCSV('titanic.csv')
data = DataCSV('../data/titanic.csv')
le = OrdinalEncoder()
data.Y_df = le.fit_transform(data.Y_df.reshape(-1, 1))


rules = Rule(
    selectors=[
        Selector(attribute=0, operator='==', value=1),
        Selector(attribute=2, operator='==', value=0),
    ], X=data.X_df, Y=data.Y_df)

