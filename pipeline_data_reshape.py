import numpy as np


for path in ['data/x_test_gaussian_noise_10.npz',
             'data/x_test_gaussian_noise_20.npz',
             'data/x_test_gaussian_noise_30.npz',
             'data/x_test_gaussian_noise_40.npz',
             'data/x_test_gaussian_noise_50.npz',
             'data/x_test_gaussian_noise_60.npz',
             'data/x_test_gaussian_noise_70.npz',
             'data/x_test_gaussian_noise_80.npz',
             'data/x_test_gaussian_noise_90.npz',
             'data/x_test_gaussian_noise_100.npz',
             'data/x_test_gaussian_noise_110.npz',
             'data/x_test_salt_pepper_noise_0.03.npz',
             'data/x_test_salt_pepper_noise_0.06.npz',
             'data/x_test_salt_pepper_noise_0.09.npz',
             'data/x_test_salt_pepper_noise_0.12.npz',
             'data/x_test_salt_pepper_noise_0.15.npz',
             'data/x_test_salt_pepper_noise_0.18.npz',
             'data/x_test_salt_pepper_noise_0.21.npz',
             'data/x_test_salt_pepper_noise_0.24.npz',
             'data/x_test_salt_pepper_noise_0.27.npz',
             'data/x_test_salt_pepper_noise_0.30.npz',
             'data/x_test_salt_pepper_noise_0.33.npz',
             'data/x_test_salt_gaussian_blur_0.5.npz',
             'data/x_test_salt_gaussian_blur_1.0.npz',
             'data/x_test_salt_gaussian_blur_1.5.npz',
             'data/x_test_salt_gaussian_blur_2.0.npz',
             'data/x_test_salt_gaussian_blur_2.5.npz',
             'data/x_test_salt_gaussian_blur_3.0.npz',
             'data/x_test_salt_gaussian_blur_3.5.npz',
             'data/x_test_salt_gaussian_blur_4.0.npz',
             'data/x_test_salt_gaussian_blur_4.5.npz',
             'data/x_test_salt_gaussian_blur_5.0.npz',
             'data/x_test_salt_gaussian_blur_5.5.npz']:
    x = np.load(path, allow_pickle=True)['data']
    print(path, x.shape, end=' ')
    x = np.reshape(x, (x.shape[0], 25088))
    print(x.shape)
