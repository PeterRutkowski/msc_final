import numpy as np
from sklearn.svm import SVC
import multiprocessing as mp
import joblib
import os
import shutil
from os import walk
from datetime import datetime


def feature_counter(path):
    counter = 0
    _, _, filenames = next(os.walk(path))
    for filename in filenames:
        if filename.startswith('model'):
            counter += 1
    return counter


def component_model(train_data):
    start_time = datetime.now()
    x, y = train_data[0], train_data[1]
    index, experiment = train_data[2], train_data[3]
    clf = SVC(kernel='poly', degree=2)
    clf.fit(x, y)

    for test_set in ['x_test_none_none',
                     'x_test_gaussian_blur_0.5',
                     'x_test_gaussian_blur_1.0',
                     'x_test_gaussian_blur_1.5',
                     'x_test_gaussian_blur_2.0',
                     'x_test_gaussian_blur_2.5',
                     'x_test_gaussian_blur_3.0',
                     'x_test_gaussian_blur_3.5',
                     'x_test_gaussian_blur_4.0',
                     'x_test_gaussian_blur_4.5',
                     'x_test_gaussian_blur_5.0',
                     'x_test_gaussian_blur_5.5']:

            x_test = np.load('pipeline_data/{}.npz'.format(test_set), allow_pickle=True)['data']

            n_features = feature_counter('pipeline_data/{}'.format(experiment))
            for i in range(n_features):
                np.savez_compressed('pipeline_data/{}/bin_rep/{}_{}'.format(experiment,
                                                                            test_set,
                                                                            index),
                                    data=np.asarray(clf.predict(x_test)).T)

    with open('pipeline_data/{}/model_{}'.format(train_data[3], train_data[2]), 'wb') as f:
        joblib.dump(clf, f, compress='lzma')
    print(datetime.now() - start_time)


experiments = ['pca60_eps100_int4',
               'pca60_eps125_int7',
               'pca60_eps150_int10',
               'pca60_eps100_int4',
               'pca60_eps125_int7',
               'pca60_eps150_int10',
               'pca60_eps100_int4',
               'pca60_eps125_int7',
               'pca60_eps150_int10']

x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']

for experiment in experiments:
    print(experiment, end=' ')
    start_time = datetime.now()
    y_train = np.load('pipeline_data/{}/bin_rep_x_train.npz'.format(experiment),
                      allow_pickle=True)['data']

    _, _, filenames = next(walk('pipeline_data/{}'.format(experiment)))

    trained = list()
    for filename in filenames:
        if filename.startswith('model'):
            trained.append(int(filename[6:]))

    to_be_trained = [n for n in range(y_train.shape[1])]

    for n in trained:
        to_be_trained.remove(n)

    try:
        os.mkdir('pipeline_data/{}/bin_rep'.format(experiment))
    except FileExistsError:
        pass

    pool = mp.Pool(70 if int(mp.cpu_count()) > 70 else mp.cpu_count())
    pool.map(component_model, [[x_train, y_train[:, i], i, experiment] for i in to_be_trained])

    end_time = datetime.now()
    print(end_time - start_time)
