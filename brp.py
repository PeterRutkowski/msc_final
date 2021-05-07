import numpy as np
import pandas as pd
from sklearn.svm import SVC
from tqdm import tqdm
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
import pickle

def component_model(x_train, y_train, x_test):
    clf = SVC(kernel='poly', degree=3)
    clf.fit(x_train, y_train)
    return clf.predict(x_test)

def BRP(exp):
    l = np.load('data/in10/in10_split_converted.npz',
                allow_pickle=True)
    x_train, x_test = l['x_train'], l['x_test_none']

    l = np.load('experiments/{}.npz'.format(exp),
              allow_pickle=True)
    y_train = l['x_train']

    y_pred = []
    for i in tqdm(range(800, y_train.shape[1])):
        prediction = component_model(x_train, y_train[:, i], x_test)
        y_pred.append(prediction)
        np.savez('experiments/BRP_{}/rep3'.format(exp), y_pred=np.asarray(y_pred).T)
        
BRP('comp120_pca_dbscan60')
