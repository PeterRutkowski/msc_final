import numpy as np

for path in ['data/x_test_gaussian_noise_10',
             'data/x_test_gaussian_noise_20',
             'data/x_test_gaussian_noise_30',
             'data/x_test_gaussian_noise_40',
             'data/x_test_gaussian_noise_50',
             'data/x_test_gaussian_noise_60',
             'data/x_test_gaussian_noise_70',
             'data/x_test_gaussian_noise_80',
             'data/x_test_gaussian_noise_90',
             'data/x_test_gaussian_noise_100',
             'data/x_test_gaussian_noise_110',
             'data/x_test_salt_pepper_noise_0.03',
             'data/x_test_salt_pepper_noise_0.06',
             'data/x_test_salt_pepper_noise_0.09',
             'data/x_test_salt_pepper_noise_0.12',
             'data/x_test_salt_pepper_noise_0.15',
             'data/x_test_salt_pepper_noise_0.18',
             'data/x_test_salt_pepper_noise_0.21',
             'data/x_test_salt_pepper_noise_0.24',
             'data/x_test_salt_pepper_noise_0.27',
             'data/x_test_salt_pepper_noise_0.30',
             'data/x_test_salt_pepper_noise_0.33']:
    x = np.load('{}.npz'.format(path), allow_pickle=True)['data']
    np.savez_compressed('{}.npz'.format(path), data=np.reshape(x, (x.shape[0], 25088)))
