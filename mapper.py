import pickle
import latent_space
import binarizer
import multiprocessing as mp
import weighted_knn
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from gtda.mapper import *


class Mapper:
    def __init__(self):
        pass

    @staticmethod
    def fit(x, projector, clusterer, n_components, n_intervals, experiment_name,
            overlap_frac=0.33, kind='uniform'):
        """Trains Mapper algorithm on x.

        :param x: Images.
        :type x: numpy.ndarray
        :param projector: Model projecting images onto latent space.
        :param clusterer: Model clustering images.
        :param n_components: Number of latent space dimensions.
        :type n_components: int
        :param n_intervals: Number of intervals covering each latent
        space component.
        :type n_intervals: int
        :param overlap_frac: Parameter defining much intervals can
        overlap.
        :type overlap_frac: float
        """

        projector = projector.fit(x)
        covering_alg = OneDimensionalCover(kind=kind,
                                           n_intervals=n_intervals,
                                           overlap_frac=overlap_frac)
        mapper_pipes = []
        
        for i in range(n_components):
            projection = Projection(columns=i)
            filter_func = Pipeline(steps=[('projector', projector), ('proj', projection)],
                                   verbose=1)
            cover = covering_alg
            mapper_pipe = make_mapper_pipeline(scaler=None,
                                               filter_func=filter_func,
                                               cover=cover,
                                               clusterer=clusterer,
                                               verbose=True,
                                               n_jobs=1)
            mapper_pipe.set_params(filter_func__proj__columns=i)
            mapper_pipes.append(('comp{}'.format(i+1), mapper_pipe))
        
        latent_projector = latent_space.LatentSpace([projector], 'latent_space')
        
        graphs = Parallel(n_jobs=int(mp.cpu_count()), prefer="threads", verbose=1)(
            delayed(mapper_pipe[1].fit_transform)(x) for mapper_pipe in mapper_pipes)
        
        covers_fitted = [covering_alg.fit(projector.transform(x)[:, i]) for i in range(n_components)]
        covers = [[(covers_fitted[j].left_limits_[i], covers_fitted[j].right_limits_[i]) for i in range(n_intervals)]
                  for j in range(n_components)]
        
        pickle.dump((latent_projector, graphs, covers), 
                    open('experiments/{}'.format(experiment_name), 'wb'))
            
    @staticmethod
    def get_representations(x_train, x_test_none, x_test_gaussian, y_train, y_test,
                            k, latent_space, graphs, covers, experiment_name):
        b = binarizer.Binarizer()
        x_train_rep = b.binarize(x_train, graphs)
        
        wknn = weighted_knn.WeightedKNN()
        x_test_none_rep = wknn.fit_transform(k, x_test_none, x_train, x_train_rep, latent_space, covers)
        x_test_gaussian_rep = wknn.fit_transform(k, x_test_gaussian, x_train, x_train_rep, latent_space, covers)
        
        np.savez('experiments/{}'.format(experiment_name), x_train=x_train_rep, 
                 x_test_none=x_test_none_rep, x_test_gaussian=x_test_gaussian_rep, 
                 y_train=y_train, y_test=y_test)
        
        
        