import numpy as np
from sklearn.svm import SVC
import multiprocessing as mp
import time
import joblib

def component_model(data):
    x_train, y_train, x_test = data[0], data[1], data[2]
    
    clf = SVC(kernel='poly', degree=2)
    clf.fit(x_train, y_train)

    joblib.dump(clf, 'data2/model_{}.{}'.format(data[3], data[4]),
                compress=('lzma', data[4]))
    
    return clf.predict(x_test)

for compression in [i for i in range(10)]:
    exp = 'comp120_dbscan60'

    start_time = time.time()
    x_train = np.load('data2/in10_split_converted.npz', allow_pickle=True)['x_train']
    x_test = np.load('data2/in10_split_converted.npz', allow_pickle=True)['x_test_none']
    y_train = np.load('data2/{}.npz'.format(exp), allow_pickle=True)['x_train']
    
    print(x_train.shape, x_test.shape, y_train.shape)

    pool = mp.Pool(4)
    data = pool.map(component_model, [[x_train, y_train[:, i], x_test, i, compression]
                                      for i in range(1)])

    f = open('data2/time_{}.txt'.format(compression), 'w')
    f.write('--- %s seconds ---' % (time.time() - start_time))
    f.close()

    x_tr = np.load('data2/{}.npz'.format(exp), allow_pickle=True)['x_train']
    y_tr = np.load('data2/{}.npz'.format(exp), allow_pickle=True)['y_train']
    y_te = np.load('data2/{}.npz'.format(exp), allow_pickle=True)['y_test']
    np.savez('data2/svc_{}'.format(exp),
             x_train=y_train, 
             x_test=np.asarray(data).T, 
             y_train=y_tr, 
             y_test=y_te)
    