import numpy as np
import mapper
import my_pca
import pickle
import os
from sklearn.cluster import DBSCAN


def mapper_training(n_components, epsilon, n_intervals):
    dir_name = 'pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals)
    os.mkdir('pipeline_data/{}'.format(dir_name))
    print(dir_name)

    clusterer = DBSCAN(eps=epsilon, min_samples=1)
    projector = my_pca.MyPCA(n_components=n_components)

    x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']

    m = mapper.Mapper()
    m.fit(x_train, projector=projector, clusterer=clusterer, n_components=n_components,
          n_intervals=n_intervals, experiment_name=dir_name, kind='uniform')

    with open('pipeline_data/{}/mapper'.format(dir_name), 'rb') as f:
        graphs = pickle.load(f)[1]

    m.get_representations(x_train, graphs, dir_name)


mapper_training(60, 90, 4)
mapper_training(60, 80, 4)
mapper_training(60, 70, 4)
