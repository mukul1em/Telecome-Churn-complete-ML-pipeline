from tkinter.filedialog import LoadFileDialog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dataset


class Model:

    def logistic():
        lr = LogisticRegression()
        return lr


class Training:
    def __init__(self, xtrain, xtest, ytrain, ytest):
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
    
    def cross_val(nsplits, nrepeats, model):
        cv = RepeatedStratifiedKFold(n_splits=nsplits, n_repeats=nrepeats, random_state=10)
        scores = cross_val_score(model, xtrain, ytrain, cv=cv, n_jobs=-1)
        return scores.mean()
    

if __name__ == '__main__':

    data = pd.read_csv('/Users/mukulrawat/Documents/ML Projects/telecom churn/Data.csv')
    data['churn'] = data['churn'].map({False:0,True:1})
    X = data.iloc[:, :-1]
    y = data.iloc[:,-1]
    X.drop('phone number', inplace=True,axis=1)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=10)
    xtrain, xtest = dataset.DatasetPrep.labelencoder(xtrain=xtrain, xtest=xtest)
    xtrain, xtest = dataset.DatasetPrep.scaling(xtrain, xtest)

    lr = Model.logistic()
    scores = Training.cross_val(10, 3, model = lr)
    print(scores)






