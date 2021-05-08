import numpy as np
from sklearn.svm import SVC
import multiprocessing as mp
import joblib
from os import walk
from datetime import datetime


def component_model(train_data):
    x, y = train_data[0], train_data[1]
    clf = SVC(kernel='poly', degree=2)
    clf.fit(x, y)

    with open('data/{}/model_{}'.format(train_data[3], train_data[2]), 'wb') as f:
        joblib.dump(clf, f, compress='zlib')


experiments = ['pca60_eps85_int4',
               'pca60_eps85_int7',
               'pca60_eps90_int4',
               'pca60_eps90_int7']

for experiment in experiments:
    print(experiment, end=' ')
    start_time = datetime.now()
    x_train = np.load('data/x_train_none_none.npz', allow_pickle=True)['in10']
    y_train = np.load('data/{}/bin_rep_x_train.npz'.format(experiment),
                      allow_pickle=True)['in10']

    _, _, filenames = next(walk('data/{}'.format(experiment)))

    trained = list()
    for filename in filenames:
        if filename.startswith('model'):
            trained.append(int(filename[6:]))

    to_be_trained = [n for n in range(y_train.shape[1])]

    for n in trained:
        to_be_trained.remove(n)

    pool = mp.Pool(70 if mp.cpu_count() > 70 else mp.cpu_count())
    pool.map(component_model, [[x_train, y_train[:, i], i, experiment] for i in to_be_trained])

    end_time = datetime.now()
    print(end_time - start_time)
