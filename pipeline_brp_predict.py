import numpy as np
import multiprocessing as mp
import joblib
from os import walk
from datetime import datetime


def predict(input_data):
    start_time = datetime.now()
    experiment, test_set = input_data[0], input_data[1]
    x = np.load('pipeline_data/{}.npz'.format(test_set), allow_pickle=True)['data']
    brp = list()

    n_features = 0
    _, _, filenames = next(walk('pipeline_data/{}'.format(experiment)))
    for filename in filenames:
        if filename.startswith('model'):
            n_features += 1

    for i in range(n_features):
        model = joblib.load('pipeline_data/{}/model_{}'.format(experiment, i))
        brp.append(model.predict(x))
        np.savez_compressed('pipeline_data/{}/brp_{}.npz'.format(experiment, test_set),
                            data=np.asarray(brp).T)

    end_time = datetime.now()
    print(experiment, test_set, end_time - start_time)


pool = mp.Pool(60)
predictions = list()
for n_components in [60]:
    for epsilon in [150, 100]:
        for n_intervals in [4, 7, 10]:
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
                predictions.append(['pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals),
                                    test_set])

pool.map(predict, predictions)
