from sklearn.svm import SVC
import numpy as np
import joblib

for svm_data in [['poly', 1], 
                 ['poly', 2], 
                 ['poly', 3]]:

    l = np.load('data2/svc_{}{}_dec_comp120_dbscan60.npz'.format(svm_data[0], svm_data[1]), 
                allow_pickle=True)
    x_train = l['x_train']
    x_test = l['x_test']
    y_train = l['y_train']
    y_test = l['y_test']

    for svm_data2 in [['sigmoid', -1], 
                     ['poly', 1], 
                     ['poly', 2], 
                     ['poly', 3]]:

        clf = SVC(kernel=svm_data2[0], degree=svm_data2[1]) if svm_data2[1] > 0 else SVC(kernel=svm_data2[0])
        clf.fit(x_train, y_train)
        print('rep {}{} classifier {}{} score {}'.format(
            svm_data[0], 
            svm_data[1],
            svm_data2[0], 
            svm_data2[1],
            clf.score(x_test, y_test)))
