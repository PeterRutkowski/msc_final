import numpy as np
import multiprocessing as mp
import joblib
import os
import shutil
from datetime import datetime


def feature_counter(path):
    counter = 0
    _, _, filenames = next(os.walk(path))
    for filename in filenames:
        if filename.startswith('model'):
            counter += 1
    return counter


def predict_feature(input_data):
    exp, index, x_test, test_name = input_data[0], input_data[1], input_data[2], input_data[3]

    with open('pipeline_data/{}/model_{}'.format(exp, index), 'rb') as f:
        np.savez_compressed('pipeline_data/{}/bin_rep/{}_{}'.format(exp, test_set, index),
                            data=np.asarray(joblib.load(f).predict(x_test)).T)


experiments = ['pca60_eps100_int4',
               'pca60_eps100_int7',
               'pca60_eps100_int10',
               'pca60_eps125_int4',
               'pca60_eps125_int7',
               'pca60_eps125_int10',
               'pca60_eps150_int4',
               'pca60_eps150_int7',
               'pca60_eps150_int10']

for experiment in experiments:
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
        start_time = datetime.now()
        if not os.path.isfile('pipeline_data/{}/bin_rep_{}.npz'.format(experiment, test_set)):
            try:
                os.mkdir('pipeline_data/{}/bin_rep'.format(experiment))
            except FileExistsError:
                shutil.rmtree('pipeline_data/{}/bin_rep'.format(experiment))
                os.mkdir('pipeline_data/{}/bin_rep'.format(experiment))

            with np.load('pipeline_data/{}.npz'.format(test_set), allow_pickle=True) as f:
                x = f['data']

            n_features = feature_counter('pipeline_data/{}'.format(experiment))
            feature_predictions = list()
            for i in range(n_features):
                feature_predictions.append([experiment, i, x, test_set])

            pool = mp.Pool(70 if int(mp.cpu_count()) > 70 else mp.cpu_count())
            pool.map(predict_feature, feature_predictions)

            bin_rep = list()
            for i in range(n_features):
                with np.load('pipeline_data/{}/bin_rep/{}_{}.npz'.format(
                        experiment, test_set, i), allow_pickle=True) as f:
                    bin_rep.append(f['data'])

            np.savez_compressed('pipeline_data/{}/bin_rep_{}'.format(experiment, test_set),
                                data=np.asarray(bin_rep).T)
            shutil.rmtree('pipeline_data/{}/bin_rep'.format(experiment))
        end_time = datetime.now()
        print(experiment,
              test_set,
              np.load('pipeline_data/{}/bin_rep_{}.npz'.format(experiment, test_set),
                      allow_pickle=True)['data'].shape,
              end_time - start_time)
