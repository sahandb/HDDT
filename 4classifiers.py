from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_Data(path, testSplit):
    dataFrame = pd.read_csv(path)
    Y = dataFrame['label']
    x = dataFrame.drop('label', 1)
    xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=testSplit, shuffle=True)

    return xTrain, xTest, yTrain, yTest


def naive(xTrain, xTest, yTrain, yTest):
    naiveBayes = GaussianNB()
    naiveBayes.fit(xTrain, yTrain)
    y_pred_naiveBayes = naiveBayes.predict(xTest)
    accuracy_naiveBayes = accuracy_score(yTest, y_pred_naiveBayes)
    return print('accuracy naive bayes: ', accuracy_naiveBayes * 100)


def onn(xTrain, xTest, yTrain, yTest):
    Onn = KNeighborsClassifier(n_neighbors=1)
    Onn.fit(xTrain, yTrain)
    y_pred_Onn = Onn.predict(xTest)
    accuracy_Onn = accuracy_score(yTest, y_pred_Onn)
    return print('accuracy ONN: ', accuracy_Onn * 100)


def linSvc(xTrain, xTest, yTrain, yTest):
    clf = LinearSVC()
    clf.fit(xTrain, yTrain)
    y_pred_svc = clf.predict(xTest)
    accuracy_svc = accuracy_score(yTest, y_pred_svc)
    return print('accuracy lin svc: ', accuracy_svc * 100)


def svmRbf(xTrain, xTest, yTrain, yTest):
    clf = svm.SVC(kernel='rbf')
    clf.fit(xTrain, yTrain)
    y_pred_svm = clf.predict(xTest)
    accuracy_svm = accuracy_score(yTest, y_pred_svm)
    return print('accuracy smv with kernel rbf: ', accuracy_svm * 100)


ConPath = './Changed_Con_Covid-19.csv'  # original file : ORG_Con_Covid-19.csv


def main():
    xTrain, xTest, yTrain, yTest = split_Data(ConPath, 0.3)
    naive(xTrain, xTest, yTrain, yTest)
    onn(xTrain, xTest, yTrain, yTest)
    linSvc(xTrain, xTest, yTrain, yTest)
    svmRbf(xTrain, xTest, yTrain, yTest)


main()
