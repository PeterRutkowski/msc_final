import numpy as np
import mapper
import my_pca
import pickle
import os
from sklearn.cluster import DBSCAN


def mapper_training(n_components, epsilon, n_intervals):
    dir_name = 'pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals)
    print(dir_name)
    os.mkdir('pipeline_data/{}'.format(dir_name))

    clusterer = DBSCAN(eps=epsilon, min_samples=1)
    projector = my_pca.MyPCA(n_components=n_components)

    x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']

    m = mapper.Mapper()
    m.fit(x_train, projector=projector, clusterer=clusterer, n_components=n_components,
          n_intervals=n_intervals, experiment_name=dir_name, kind='uniform')

    graphs = pickle.load(open('pipeline_data/{}/mapper'.format(dir_name), 'rb'))[1]
    m.get_representations(x_train, graphs, dir_name)


#mapper_training(120, 200, 4)
#mapper_training(120, 200, 7)
#mapper_training(120, 200, 10)
#mapper_training(90, 200, 4)
#mapper_training(90, 200, 7)
#mapper_training(90, 200, 10)
#mapper_training(60, 200, 4)
#mapper_training(60, 200, 7)
#mapper_training(60, 200, 10)
mapper_training(120, 150, 4)
mapper_training(120, 150, 7)
mapper_training(120, 150, 10)
mapper_training(90, 150, 4)
mapper_training(90, 150, 7)
mapper_training(90, 150, 10)
mapper_training(60, 150, 4)
mapper_training(60, 150, 7)
mapper_training(60, 150, 10)
