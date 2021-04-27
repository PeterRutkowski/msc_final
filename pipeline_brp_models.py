import numpy as np
from sklearn.svm import SVC
import multiprocessing as mp
import joblib
from os import walk


def component_model(train_data):
    x, y = train_data[0], train_data[1]
    clf = SVC(kernel='poly', degree=2)
    clf.fit(x, y)
 
    joblib.dump(clf, 'pipeline_data/{}/model_{}'.format(train_data[3], train_data[2]),
                compress='lzma')
    print(train_data[2], end=' ')


experiments = list()
for n_components in [60, 90, 120]:
    for epsilon in [150, 100]:
        for n_intervals in [4, 7, 10]:
            experiments.append('pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals))

for experiment in experiments:
    print(experiment)
    x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']
    y_train = np.load('pipeline_data/{}/brp_x_train.npz'.format(experiment),
                      allow_pickle=True)['data']

    _, _, filenames = next(walk('pipeline_data/{}'.format(experiment)))

    trained = list()
    for filename in filenames:
        if filename.startswith('model'):
            trained.append(int(filename[6:]))

    to_be_trained = [n for n in range(y_train.shape[1])]

    for n in trained:
        to_be_trained.remove(n)

    pool = mp.Pool(int(mp.cpu_count()))
    pool.map(component_model, [[x_train, y_train[:, i], i, experiment] for i in to_be_trained])
