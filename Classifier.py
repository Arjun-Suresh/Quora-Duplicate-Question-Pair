import sys
sys.path.append('//Feature_Extracter.py')
from Feature_Extracter import feature_extracter
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
import xgboost as xgb
import matplotlib.pyplot as plt


def naive_bayes_classifier(x_train, y_train, x_valid, y_valid):
    clf=MultinomialNB()
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_valid)
    k=0
    for i in range(len(y_valid)):
        if y_valid.data[i] == y_pred[i]:
            k+=1
    return (float(k)/float(len(y_valid)))


def xgboost_classifier(x_train, y_train, x_valid, y_valid):
    params = {}

    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'logloss'
    params["eta"] = 0.02
    params["subsample"] = 0.7
    params["min_child_weight"] = 1
    params["colsample_bytree"] = 0.7
    params["max_depth"] = 4
    params["silent"] = 1
    params["seed"] = 1632

    d_train = xgb.DMatrix(x_train, label=y_train)
    d_valid = xgb.DMatrix(x_valid, label=y_valid)
    watchlist = [(d_train, 'train')]
    bst = xgb.train(params, d_train, 500, watchlist, early_stopping_rounds=50, verbose_eval=100)
    sub = pd.DataFrame()
    sub['is_duplicate'] = bst.predict(d_valid)
    #plt.rcParams['figure.figsize'] = (7.0, 7.0)
    #xgb.plot_importance(bst)
    #plt.show()
    k = 0
    for i in range(len(y_valid)):
        if y_valid.data[i] == int(sub["is_duplicate"][i]+0.5):
            k += 1
    return (float(k) / float(len(y_valid)))


def svm_classifier(x_train, y_train, x_valid, y_valid):
    clf = svm.SVC(C=1000)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_valid)
    k = 0
    for i in range(len(y_valid)):
        if y_valid.data[i] == y_pred[i]:
            k += 1
    return(float(k) / float(len(y_valid)))


def rebalanceClasses(train):
    pos_train = train[train['is_duplicate'] == 1]
    neg_train = train[train['is_duplicate'] == 0]
    p = 0.165
    scale = ((len(pos_train) / (len(pos_train) + len(neg_train))) / p) - 1
    while scale > 1:
        neg_train = pd.concat([neg_train, neg_train])
        scale -= 1
    neg_train = pd.concat([neg_train, neg_train[:int(scale * len(neg_train))]])
    train = pd.concat([pos_train, neg_train])
    return train


def main(train_file, classifierType):
    featureExtracter = feature_extracter(train_file)
    train = pd.read_csv(train_file)
    train = featureExtracter.get_features(train)
    col = [c for c in train.columns if c[:1] == 'z']
    train = rebalanceClasses(train)
    x_train, x_valid, y_train, y_valid = train_test_split(train[col], train['is_duplicate'], test_size=0.4, random_state=0)
    if int(classifierType) == 1:
        accuracy = naive_bayes_classifier(x_train, y_train, x_valid, y_valid)
    elif int(classifierType) == 3:
        accuracy = svm_classifier(x_train, y_train, x_valid, y_valid)
    elif int(classifierType) == 2:
        accuracy = xgboost_classifier(x_train, y_train, x_valid, y_valid)
    print (accuracy)


if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage:\tClassifier.py <train_file> <classifierType (1 for Naive Bayes, 2 for XGBoost, 3 for SVM)>')
        sys.exit(0)
    main(sys.argv[1],sys.argv[2])