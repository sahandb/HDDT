import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
import copy
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc


# classes
class Tree:
    def __init__(self):
        # var
        self.tree = {}

    # HDDT func
    def hddt(self, data, root, cutOff, mHeight, remainingFeatures, depth=0):
        dataFrame = deepcopy(data)
        root['depth'] = depth
        labels = list(dict.fromkeys(dataFrame['label'].values))
        # if 1 return the class if 0 mean cant classified and else NONE
        if len(labels) == 1:
            root['label'] = labels[0]
            return root
        elif len(labels) == 0:
            root['label'] = 'Not-Classified'
            return root
        else:
            root['label'] = None

        labels = list(dict.fromkeys(dataFrame['label'].values))
        # pruning, majority
        if ((dataFrame.shape[0] < cutOff) or (mHeight <= depth)):
            if (list(dataFrame['label'].values).count(labels[0]) >= list(dataFrame['label'].values).count(labels[1])):
                root['label'] = labels[0]
            else:
                root['label'] = labels[1]
            return root
        biHell = self.bi_Hellinger(dataFrame, remainingFeatures)
        root['distance'] = biHell['distance']
        root['value'] = biHell['value']
        root['feature'] = biHell['feature']
        root['index'] = biHell['index']

        remainingFeatures.append(biHell['feature'])

        treeLeft = dataFrame.drop(dataFrame[dataFrame[biHell['feature']] > biHell['value']].index, axis=0)
        treeRight = dataFrame.drop(dataFrame[dataFrame[biHell['feature']] <= biHell['value']].index, axis=0)
        print('====')
        print(root)
        print('[ data Left:{0} ] -- [ all:{1} ] -- [ data right: {2} ]'.format(treeLeft.shape[0], dataFrame.shape[0],
                                                                               treeRight.shape[0]))

        root['Left_Child'] = self.hddt(treeLeft, {}, cutOff, mHeight, deepcopy(remainingFeatures), depth + 1)
        root['Right_Child'] = self.hddt(treeRight, {}, cutOff, mHeight, deepcopy(remainingFeatures), depth + 1)

        return root

    def bi_Hellinger(self, dataB, nremainingFeatures):
        dataFrame = deepcopy(dataB)
        # all of the values initial with -1
        helDictionary = {'distance': -1, 'feature': -1, 'value': -1, 'index': -1}
        features = list(dataFrame.columns.values)
        labels = list(dict.fromkeys(dataFrame['label'].values))

        # extract positive and negative classes
        positiveClass = dataFrame.drop(dataFrame[dataFrame['label'] == min(labels)].index, axis=0)
        negativeClass = dataFrame.drop(dataFrame[dataFrame['label'] == max(labels)].index, axis=0)

        # delete visited feature
        features.remove('label')
        for idx in nremainingFeatures:
            features.remove(idx)

        # calculate
        for feature in features:
            # Tf in algorithm
            fList = list(dict.fromkeys(dataFrame[feature].values))
            fList.sort()
            for f in fList:
                # where f = feature and class is positive
                zfPositive = positiveClass.where(positiveClass[feature] == f).dropna().shape[0]
                # where f = feature and class is negative
                zfNegative = negativeClass.where(negativeClass[feature] == f).dropna().shape[0]
                # where f != feature and class is positive
                zfpPositive = positiveClass.where(positiveClass[feature] != f).dropna().shape[0]
                # where f != feature and class is negative
                zfpNegative = negativeClass.where(negativeClass[feature] != f).dropna().shape[0]

                # Hellinger formula
                hellingerDistance = (np.sqrt(zfPositive / positiveClass.shape[0]) - np.sqrt(zfNegative / negativeClass.shape[0])) ** 2 + (np.sqrt(zfpPositive / positiveClass.shape[0]) - np.sqrt(zfpNegative / negativeClass.shape[0])) ** 2

                if hellingerDistance > helDictionary['distance']:
                    helDictionary['distance'] = hellingerDistance
                    helDictionary['value'] = f
                    helDictionary['feature'] = feature
                    helDictionary['index'] = features.index(feature)
        # for check error if any -1 be in dictionary and print it else sqrt distance and return
        if helDictionary['distance'] < 0:
            print(helDictionary['distance'])
        helDictionary['distance'] = np.sqrt(helDictionary['distance'])
        return helDictionary

    def train(self, dataf, cutOff, mHeight=-1):
        return self.hddt(dataf, self.tree, cutOff, mHeight, [])

    def seen(self, s, tree):
        if tree['label'] == 'Not-Classified':
            return -1
        elif tree['label'] != None:
            return tree['label']

        elif s[tree['index']] <= tree['value']:
            return self.seen(s, tree['Left_Child'])

        elif s[tree['index']] > tree['value']:
            return self.seen(s, tree['Right_Child'])

    def predict_Class(self, datap):
        predicted = list()
        for pr in datap.values:
            seen = self.seen(pr, self.tree)
            predicted.append(seen)
        return predicted


# read data and train test split 70 ., 30
def split_Data(path, testSplit):
    dataFrame = pd.read_csv(path)
    Y = dataFrame['label']
    x = dataFrame.drop('label', 1)
    xTrain, xTest, yTrain, yTest = train_test_split(x, Y, test_size=testSplit, shuffle=True)

    table = pd.concat([xTrain, yTrain], axis=1)
    return table, xTest, yTest


# OVA
def ova(dataova, label):
    daataa = dataova.copy()
    daataa['label'] = 1 * (daataa['label'] == label)
    return daataa


# OVO
def ovo(dataovo, deleteLabel):
    daataa = dataovo.copy()
    daataa.drop(daataa[daataa['label'] == deleteLabel].index, axis=0, inplace=True)
    return daataa


def save(name, maxHeight, c, report, iterr):
    file = open(name + '.txt', 'a')
    file.writelines('c is : {0} and max height is : {1}\n'.format(c, maxHeight))
    report = np.array(report)
    report = report / iterr
    file.writelines(name + ' performance is:')
    file.writelines(
        '\n Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4} \n'.format(report[0], report[1],
                                                                                          report[2], report[3],
                                                                                          report[4]))


def runOVA(iter, cutOff, maxHeight, dataTrain, dataTest, yTest):
    reportValues = []
    for it in range(iter):
        accuracyOVA = 0
        # Create trees
        tree0 = Tree()
        tree1 = Tree()
        tree2 = Tree()

        # FILL DATA
        d0 = ova(dataTrain, 0)
        d1 = ova(dataTrain, 1)
        d2 = ova(dataTrain, 2)

        # TRAIN
        print(':::::::::::::::::: tree 0 ova ::::::::::::::::::')
        print(tree0.train(d0, cutOff, maxHeight))
        print(':::::::::::::::::: tree 1 ova ::::::::::::::::::')
        print(tree1.train(d1, cutOff, maxHeight))
        print(':::::::::::::::::: tree 2 ova ::::::::::::::::::')
        print(tree2.train(d2, cutOff, maxHeight))

        # TEST
        predict0 = tree0.predict_Class(dataTest)
        predict1 = tree1.predict_Class(dataTest)
        predict2 = tree2.predict_Class(dataTest)

        for y in range(len(yTest.values)):
            if yTest.values[y] == 0 and predict0[y] == 1 and predict1[y] == 0 and predict2[y] == 0:
                accuracyOVA += 1
            elif yTest.values[y] == 1 and predict0[y] == 0 and predict1[y] == 1 and predict2[y] == 0:
                accuracyOVA += 1
            elif yTest.values[y] == 2 and predict0[y] == 0 and predict1[y] == 0 and predict2[y] == 1:
                accuracyOVA += 1
        # minor class reports
        convert_y_2to1 = 1 * (yTest.values == 2)
        report = prfs(convert_y_2to1, predict2, average='binary')
        fpr, tpr, thresh = roc_curve(convert_y_2to1, predict2)
        report_auc = auc(fpr, tpr) * 100
        report_accuracy = accuracyOVA * 100 / len(yTest.values)

        print('\n ova \n Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4} \n'.format(report[0] * 100,
                                                                                                       report[1] * 100,
                                                                                                       report[2] * 100,
                                                                                                       report_auc,
                                                                                                       report_accuracy))

        reportValues.append(report[0] * 100)
        reportValues.append(report[1] * 100)
        reportValues.append(report[2] * 100)
        reportValues.append(report_auc)
        reportValues.append(report_accuracy)
        save("OVA", maxHeight, cutOff, reportValues, iter)


def runOVO(iter, cutOff, maxHeight, dataTrain, dataTest, yTest):
    reportValues = []
    for it in range(iter):
        accuracyOVO = 0

        # Create trees
        tree01 = Tree()
        tree12 = Tree()
        tree02 = Tree()

        # FILL DATA
        d01 = ovo(dataTrain, 2)
        d12 = ovo(dataTrain, 0)
        d02 = ovo(dataTrain, 1)

        # TRAIN
        print(':::::::::::::::::: tree 0 1 ovo ::::::::::::::::::')
        print(tree01.train(d01, cutOff, maxHeight))
        print(':::::::::::::::::: tree 1 2 ovo ::::::::::::::::::')
        print(tree12.train(d12, cutOff, maxHeight))
        print(':::::::::::::::::: tree 0 2 ovo ::::::::::::::::::')
        print(tree02.train(d02, cutOff, maxHeight))

        # TEST
        predict01 = tree01.predict_Class(dataTest)
        predict12 = tree12.predict_Class(dataTest)
        predict02 = tree02.predict_Class(dataTest)

        class2 = []
        for y in range(len(yTest.values)):
            if yTest.values[y] == 0 and predict01[y] == 0 and predict02[y] == 0:
                accuracyOVO += 1
                class2.append(0)
            elif yTest.values[y] == 1 and predict01[y] == 1 and predict12[y] == 1:
                accuracyOVO += 1
                class2.append(0)
            elif yTest.values[y] == 2 and predict12[y] == 2 and predict02[y] == 2:
                accuracyOVO += 1
                class2.append(1)

            elif yTest.values[y] == 0 and (predict01[y] == 0 and predict02[y] == 0):
                class2.append(0)
            elif yTest.values[y] == 1 and (predict01[y] == 1 and predict12[y] == 1):
                class2.append(0)
            elif yTest.values[y] == 2 and (predict12[y] == 2 and predict02[y] == 2):
                class2.append(0)
            else:
                class2.append(0)

        # minor class reports
        convert_y_2to1 = 1 * (yTest.values == 2)
        report = prfs(convert_y_2to1, class2, average='binary')
        fpr, tpr, thresh = roc_curve(convert_y_2to1, class2)
        report_auc = auc(fpr, tpr) * 100
        report_accuracy = accuracyOVO * 100 / len(yTest.values)

        print('\n ovo \n Precision = {0}, Recall = {1}, F1 = {2}, AUC = {3}, Accuracy = {4} \n'.format(report[0] * 100,
                                                                                                       report[1] * 100,
                                                                                                       report[2] * 100,
                                                                                                       report_auc,
                                                                                                       report_accuracy))

        reportValues.append(report[0] * 100)
        reportValues.append(report[1] * 100)
        reportValues.append(report[2] * 100)
        reportValues.append(report_auc)
        reportValues.append(report_accuracy)
        save("OVO", maxHeight, cutOff, reportValues, iter)


# paths
DisPath = './Changed_Dis_Covid-19.csv'  # original file : ORG_Dis_Covid-19.csv
ConPath = './Changed_Con_Covid-19.csv'  # original file : ORG_Con_Covid-19.csv
cutOffs = [10,50,500]
maxHeights = [2,3,4,5]


def main():
    dataTrain, dataTest, yTest = split_Data(DisPath, 0.3)
    for cutoff in cutOffs:
        for maxHeight in maxHeights:
            runOVA(10, cutoff, maxHeight, dataTrain, dataTest, yTest)
            runOVO(10, cutoff, maxHeight, dataTrain, dataTest, yTest)


main()
