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
    model = joblib.load('pipeline_data/{}/model_{}'.format(exp, index))
    np.savez_compressed('pipeline_data/{}/bin_rep/{}_{}'.format(exp, test_set, index),
                        data=np.asarray(model.predict(x_test)).T)


experiments = list()
for n_components in [60, 90, 120]:
    for epsilon in [150, 100]:
        for n_intervals in [4, 7, 10]:
            experiments.append('pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals))


experiments = ['pca15_eps150_int4',
               'pca30_eps150_int4',
               'pca45_eps150_int4']

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
                     'x_test_gaussian_blur_5.5',
                     'x_test_gaussian_noise_10',
                     'x_test_gaussian_noise_20',
                     'x_test_gaussian_noise_30',
                     'x_test_gaussian_noise_40',
                     'x_test_gaussian_noise_50',
                     'x_test_gaussian_noise_60',
                     'x_test_gaussian_noise_70',
                     'x_test_gaussian_noise_80',
                     'x_test_gaussian_noise_90',
                     'x_test_gaussian_noise_100',
                     'x_test_gaussian_noise_110',
                     'x_test_salt_pepper_noise_0.03',
                     'x_test_salt_pepper_noise_0.06',
                     'x_test_salt_pepper_noise_0.09',
                     'x_test_salt_pepper_noise_0.12',
                     'x_test_salt_pepper_noise_0.15',
                     'x_test_salt_pepper_noise_0.18',
                     'x_test_salt_pepper_noise_0.21',
                     'x_test_salt_pepper_noise_0.24',
                     'x_test_salt_pepper_noise_0.27',
                     'x_test_salt_pepper_noise_0.30',
                     'x_test_salt_pepper_noise_0.33']:
        start_time = datetime.now()
        if not os.path.isfile('pipeline_data/{}/bin_rep_{}.npz'.format(experiment, test_set)):
            try:
                os.mkdir('pipeline_data/{}/bin_rep'.format(experiment))
            except FileExistsError:
                shutil.rmtree('pipeline_data/{}/bin_rep'.format(experiment))
                os.mkdir('pipeline_data/{}/bin_rep'.format(experiment))
            x = np.load('pipeline_data/{}.npz'.format(test_set), allow_pickle=True)['data']

            n_features = feature_counter('pipeline_data/{}'.format(experiment))
            feature_predictions = list()
            for i in range(n_features):
                feature_predictions.append([experiment, i, x, test_set])

            pool = mp.Pool(50)
            pool.map(predict_feature, feature_predictions)

            bin_rep = list()
            for i in range(n_features):
                bin_rep.append(np.load('pipeline_data/{}/bin_rep/{}_{}.npz'.format(
                    experiment, test_set, i), allow_pickle=True)['data'])
            np.savez_compressed('pipeline_data/{}/bin_rep_{}'.format(experiment, test_set),
                                data=np.asarray(bin_rep).T)
            shutil.rmtree('pipeline_data/{}/bin_rep'.format(experiment))
        end_time = datetime.now()
        print(experiment,
              test_set,
              np.load('pipeline_data/{}/bin_rep_{}.npz'.format(experiment, test_set),
                      allow_pickle=True)['data'].shape,
              end_time - start_time)
