import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


class DatasetPrep:
    def __init__(self, xtrain, xtest):
        self.xtrain = xtrain
        self.xtest = xtest
    def labelencoder(xtrain, xtest): 
        for col in xtrain.select_dtypes('object').columns:
            lb = LabelEncoder()
            lb.fit(xtrain[col])
            xtest[col] = lb.transform(xtest[col])
            xtrain[col] = lb.transform(xtrain[col])
        return xtrain, xtest
    def scaling(xtrain, xtest):
        for col in xtrain.columns:
            sc = StandardScaler()
            sc.fit(xtrain[col].values.reshape(-1,1))
            xtrain[col] = sc.transform(xtrain[col].values.reshape(-1,1))
            xtest[col] = sc.transform(xtest[col].values.reshape(-1,1))
        return xtrain, xtest
            

    