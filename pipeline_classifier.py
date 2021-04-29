from sklearn.svm import SVC
import numpy as np
import torch
import pickle
import pandas as pd

experiments = list()
for n_components in [60, 90, 120]:
    for epsilon in [150, 100]:
        for n_intervals in [4, 7, 10]:
            experiments.append('pca{}_eps{}_int{}'.format(n_components, epsilon, n_intervals))

experiments = ['pca15_eps150_int4',
               'pca30_eps150_int4',
               'pca45_eps150_int4']

for experiment in experiments:
    x_train = np.load('pipeline_data/{}/bin_rep_x_train.npz'.format(experiment),
                      allow_pickle=True)['data']
    y_train = np.load('pipeline_data/y_train.npz', allow_pickle=True)['data']
    y_test = np.load('pipeline_data/y_test.npz', allow_pickle=True)['data']

    try:
        clf = pickle.load(open('pipeline_data/{}/classifier'.format(experiment), 'rb'))
    except FileNotFoundError:
        clf = SVC(kernel='poly', degree=2)
        clf.fit(x_train, y_train)
        pickle.dump(clf, open('pipeline_data/{}/classifier'.format(experiment), 'wb'))

    nn = torch.load('pipeline_data/nn_benchmark.pt', map_location=torch.device('cpu'))

    scores = list()

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
        print(test_set)
        x_test = np.load('pipeline_data/{}/bin_rep_{}.npz'.format(experiment, test_set),
                         allow_pickle=True)['data']

        scores.append([' '.join(test_set.split('_')[2:-1]),
                       'Mapper classifier',
                       0.0 if test_set.split('_')[-1] == 'none' else test_set.split('_')[-1],
                       clf.score(x_test, y_test)])

        nn_x_test = torch.Tensor(np.load('pipeline_data/{}.npz'.format(test_set),
                                         allow_pickle=True)['data'])
        nn_y_test = np.squeeze(torch.LongTensor(y_test))

        nn.eval()
        outputs = nn(nn_x_test)
        _, predicted = torch.max(outputs, 1)
        eval_mask = (predicted == nn_y_test).squeeze()
        eval_score = eval_mask.sum().item()

        scores.append([' '.join(test_set.split('_')[2:-1]),
                       'VGG benchmark',
                       0.0 if test_set.split('_')[-1] == 'none' else test_set.split('_')[-1],
                       np.round(eval_score / x_test.shape[0], 3)])

    pickle.dump(pd.DataFrame(scores, columns=['noise', 'model', 'noise scale', 'accuracy']),
                open('pipeline_data/{}/scores'.format(experiment), 'wb'))
