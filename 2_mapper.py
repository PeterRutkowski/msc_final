import numpy as np
import mapper
import my_umap
import my_pca
import pickle
from sklearn.cluster import DBSCAN
from gtda.mapper import FirstSimpleGap

experiment_name = 'comp120_dbscan60'
epsilon = 60
n_components = 120
clusterer = DBSCAN(eps=epsilon, min_samples=1)
projector = my_pca.MyPCA(n_components=n_components)
    
loaded = np.load('data2/in10_split_converted.npz', allow_pickle=True)
x_train = loaded['x_train']
x_test_none = loaded['x_test_none']
y_train = loaded['y_train']
y_test = loaded['y_test']
    
print(experiment_name)
print(x_train.shape, x_test_none.shape, y_train.shape, y_test.shape)

m = mapper.Mapper()
'''m.fit(x_train, projector=projector, clusterer=clusterer, n_components=n_components, 
          n_intervals=10, experiment_name=experiment_name, kind='uniform')'''

mapper_data = pickle.load(open('data2/{}'.format(experiment_name), 'rb'))
latent_space, graphs, covers = mapper_data[0], mapper_data[1], mapper_data[2]
k = 5
    
m.get_representations(x_train, x_test_none, [], y_train, y_test,
                          k, latent_space, graphs, covers, experiment_name)
    
loaded = np.load('data2/dec_{}.npz'.format(experiment_name), allow_pickle=True)
x_train = loaded['x_train']
x_test_none = loaded['x_test_none']
y_train = loaded['y_train']
y_test = loaded['y_test']
print(x_train.shape, x_test_none.shape, y_train.shape, y_test.shape)
