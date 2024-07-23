import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin

class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract fist letter of variable

    def __init__(self, variable):
        if variable is None:
            raise ValueError("please provide a variable to extract letter")
        self.variable = variable

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        first_letter = []
        for row in X[self.variable]:
            try:
                first_letter.append(str(row)[0])
            except:
                first_letter.append(row)

        X[self.variable+'_letter'] = first_letter

        return X
    