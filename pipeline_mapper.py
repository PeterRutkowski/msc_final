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

    mapper.fit(x=np.load('data/x_train_none_none.npz', allow_pickle=True)['data'],
               projector=my_pca.MyPCA(n_components=n_components),
               clusterer=DBSCAN(eps=epsilon, min_samples=1),
               n_components=n_components,
               n_intervals=n_intervals,
               experiment_name=dir_name,
               save_path='data',
               kind='uniform')

    with open('data/{}/mapper'.format(dir_name), 'rb') as f:
        graphs = pickle.load(f)[1]
    print(datetime.now() - start_time)
    mapper.get_representations(x=np.load('data/x_train_none_none.npz', allow_pickle=True)['data'],
                               graphs=graphs,
                               experiment_name=dir_name,
                               save_path='data')


mapper_training(60, 120, 4)
mapper_training(60, 120, 7)
