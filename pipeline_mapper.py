import numpy as np
import mapper
import my_pca
import pickle
from sklearn.cluster import DBSCAN

epsilon = 200
n_components = 240
experiment_name = 'comp{}_dbscan{}'.format(n_components, epsilon)

clusterer = DBSCAN(eps=epsilon, min_samples=1)
projector = my_pca.MyPCA(n_components=n_components)

x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']

m = mapper.Mapper()
m.fit(x_train, projector=projector, clusterer=clusterer, n_components=n_components,
      n_intervals=10, experiment_name=experiment_name, kind='uniform')

graphs = pickle.load(open('pipeline_data/mapper_{}'.format(experiment_name), 'rb'))[1]
m.get_representations(x_train, graphs, experiment_name)
