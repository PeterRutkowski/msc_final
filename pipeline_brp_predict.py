import numpy as np
import multiprocessing as mp
import joblib


def predict(test_set):
    x = np.load('pipeline_data/{}.npz'.format(test_set), allow_pickle=True)['data']
    brp = list()

    for i in range(1200):
        print(i)
        model = joblib.load('pipeline_data/comp120_dbscan200/model_{}'.format(i))
        brp.append(model.predict(x))

    np.savez_compressed('pipeline_data/comp120_dbscan200/rep_{}.npz'.format(test_set),
                        data=np.asarray(brp).T)


pool = mp.Pool(34)
pool.map(predict, ['x_test_none_none',
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
