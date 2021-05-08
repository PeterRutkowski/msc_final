import pickle
import latent_space
import binarizer
import multiprocessing as mp
import numpy as np
from joblib import Parallel, delayed
from sklearn.pipeline import Pipeline
from gtda.mapper import *


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
    mapper_pipes = []

    for i in range(n_components):
        mapper_pipe = make_mapper_pipeline(scaler=None,
                                           filter_func=Pipeline(
                                               steps=[('projector', projector),
                                                      ('proj', Projection(columns=i))],
                                               verbose=1),
                                           cover=OneDimensionalCover(kind=kind,
                                                                     n_intervals=n_intervals,
                                                                     overlap_frac=overlap_frac),
                                           clusterer=clusterer,
                                           verbose=False,
                                           n_jobs=1)
        mapper_pipe.set_params(filter_func__proj__columns=i)
        mapper_pipes.append(('comp{}'.format(i+1), mapper_pipe))

    latent_projector = latent_space.LatentSpace([projector], 'latent_space')

    graphs = Parallel(n_jobs=int(0.8*mp.cpu_count()), prefer="threads", verbose=1)(
        delayed(mapper_pipe[1].fit_transform)(x) for mapper_pipe in mapper_pipes)

    x_proj = projector.transform(x)
    covers = []
    for i in range(n_components):
        odc = OneDimensionalCover(kind=kind,
                                  n_intervals=n_intervals,
                                  overlap_frac=overlap_frac).fit(x_proj[:, i])
        covers.append([(odc.left_limits_[j], odc.right_limits_[j])
                       for j in range(n_intervals)])

    with open('pipeline_data/{}/mapper'.format(experiment_name), 'wb') as f:
        pickle.dump((latent_projector, graphs, covers), f)


def get_representations(x, graphs, experiment_name):
    b = binarizer.Binarizer()
    x_rep = b.binarize(x, graphs)
    print(x_rep.shape)
    np.savez_compressed('pipeline_data/{}/bin_rep_x_train.npz'.format(experiment_name),
                        data=x_rep)
