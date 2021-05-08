import numpy as np
import mapper
import my_pca
import pickle
import os
from sklearn.cluster import DBSCAN
from datetime import datetime


def mapper_training(n_components, epsilon, n_intervals):
    start_time = datetime.now()
    dir_name = 'pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals)
    os.mkdir('data/{}'.format(dir_name))
    print(dir_name, end=' ')

    clusterer = DBSCAN(eps=epsilon, min_samples=1)
    projector = my_pca.MyPCA(n_components=n_components)

    x_train = np.load('data/x_train_none_none.npz', allow_pickle=True)['data']

    m = mapper.Mapper()
    m.fit(x_train, projector=projector, clusterer=clusterer, n_components=n_components,
          n_intervals=n_intervals, experiment_name=dir_name, kind='uniform')

    with open('data/{}/mapper'.format(dir_name), 'rb') as f:
        graphs = pickle.load(f)[1]
    print(datetime.now() - start_time)
    m.get_representations(x_train, graphs, dir_name)


mapper_training(10, 100, 4)
