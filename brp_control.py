import numpy as np
import joblib
import multiprocessing as mp
from os import walk
import time

'''for test_set in ['x_test_none_none',
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
    print(test_set,
          np.load('pipeline_data/comp120_dbscan200/rep_{}.npz'.format(test_set),
                  allow_pickle=True)['data'].shape)'''


def bad(input_data):
    ind, exp = input_data[0], input_data[1]
    try:
        joblib.load('pipeline_data/{}/model_{}'.format(exp, ind))
        return -1
    except EOFError:
        return ind


experiments = list()
for n_components in [60, 90, 120]:
    for epsilon in [150, 100]:
        for n_intervals in [4, 7, 10]:
            experiments.append('pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals))

for experiment in experiments:
    print(experiment)
    print(time.strftime("%H:%M:%S", time.localtime()))
    _, _, filenames = next(walk('pipeline_data/{}'.format(experiment)))
    trained = list()
    for filename in filenames:
        if filename.startswith('model'):
            trained.append([int(filename[6:]), experiment])
    print(len(trained))
    pool = mp.Pool(50)
    result = pool.map(bad, trained)
    print(time.strftime("%H:%M:%S", time.localtime()))
    print(list(np.unique(result)))
    print()


'''def combine(test_set):
    a = np.load('pipeline_data/comp120_dbscan200/rep_{}.npz'.format(test_set),
                allow_pickle=True)['data']
    b = np.load('pipeline_data/comp120_dbscan200/rep_{}_2.npz'.format(test_set),
                allow_pickle=True)['data']

    c = np.concatenate((a, b), axis=1)

    np.savez_compressed('pipeline_data/comp120_dbscan200/br_{}'.format(test_set),
                        data=c)

    print(test_set, c.shape)


pool = mp.Pool(int(mp.cpu_count()))
result = pool.map(combine, ['x_test_none_none',
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
                            'x_test_salt_pepper_noise_0.33'])
'''