from sklearn.svm import SVC
import numpy as np

x_train = np.load('pipeline_data/comp120_dbscan60/rep_x_train_none_none.npz',
                  allow_pickle=True)['data']
y_train = np.load('pipeline_data/y_train.npz', allow_pickle=True)['data']
y_test = np.load('pipeline_data/y_test.npz', allow_pickle=True)['data']

for degree in [1, 2, 3]:
    clf = SVC(kernel='poly', degree=degree)
    clf.fit(x_train, y_train)

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
        score = clf.score(np.load('pipeline_data/comp120_dbscan60/rep_{}.npz'.format(test_set),
                                  allow_pickle=True)['data'], y_test)
        print(degree, test_set, score)
