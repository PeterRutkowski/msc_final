from sklearn.svm import SVC
import numpy as np
import torch
import pickle
import pandas as pd
import plotly.express as px
import os


def plot(experiment_name, noise_type):
    df = pickle.load(open('data/{}/scores'.format(experiment_name), 'rb'))
    fig = px.line(df[df['noise'].isin([noise_type, 'none'])],
                  x='noise scale',
                  y='accuracy',
                  color='model',
                  title='{} {} {}'.format(experiment_name,
                                          noise_type,
                                          np.load('data/{}/bin_rep_x_train.npz'.format(experiment),
                                                  allow_pickle=True)['data'].shape[1]),
                  range_y=[0, 1])
    return fig


experiments = ['pca60_eps120_int4',
               'pca60_eps100_int4',
               'pca60_eps90_int4',
               'pca60_eps120_int7',
               'pca60_eps100_int7',
               'pca60_eps90_int7']

try:
    os.remove('data/plots.html')
except FileNotFoundError:
    pass


for exp_index, experiment in enumerate(experiments):
    if not os.path.isfile('data/{}/scores'.format(experiment)):
        x_train = np.load('data/{}/bin_rep_x_train.npz'.format(experiment),
                          allow_pickle=True)['data']
        y_train = np.load('data/y_train.npz', allow_pickle=True)['data']
        y_test = np.load('data/y_test.npz', allow_pickle=True)['data']

        try:
            clf = pickle.load(open('data/{}/classifier'.format(experiment), 'rb'))
        except FileNotFoundError:
            clf = SVC(kernel='poly', degree=2)
            clf.fit(x_train, y_train)
            pickle.dump(clf, open('data/{}/classifier'.format(experiment), 'wb'))

        nn = torch.load('data/benchmark_f.pt', map_location=torch.device('cpu')).eval()

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
            x_test = np.load('data/{}/bin_rep_{}.npz'.format(experiment, test_set),
                             allow_pickle=True)['data']

            scores.append([' '.join(test_set.split('_')[2:-1]),
                           'Mapper classifier',
                           0.0 if test_set.split('_')[-1] == 'none' else test_set.split('_')[-1],
                           clf.score(x_test, y_test)])

            nn_x_test = torch.Tensor(np.load('data/{}.npz'.format(test_set),
                                             allow_pickle=True)['data'])
            nn_y_test = np.squeeze(torch.LongTensor(y_test))

            outputs = nn(nn_x_test)
            _, predicted = torch.max(outputs, 1)
            eval_mask = (predicted == nn_y_test).squeeze()
            eval_score = eval_mask.sum().item()

            scores.append([' '.join(test_set.split('_')[2:-1]),
                           'VGG benchmark',
                           0.0 if test_set.split('_')[-1] == 'none' else test_set.split('_')[-1],
                           np.round(eval_score / x_test.shape[0], 3)])

            print(experiment, test_set)

        pickle.dump(pd.DataFrame(scores,
                                 columns=['noise', 'model', 'noise scale', 'accuracy']),
                    open('data/{}/scores'.format(experiment), 'wb'))


with open('data/plots.html', 'a') as f:
    for experiment in experiments:
        f.write(plot(experiment, 'gaussian blur').to_html(full_html=False, include_plotlyjs='cdn'))
    for experiment in experiments:
        f.write(plot(experiment, 'gaussian noise').to_html(full_html=False, include_plotlyjs='cdn'))
    for experiment in experiments:
        f.write(plot(experiment, 'salt pepper noise').to_html(full_html=False, include_plotlyjs='cdn'))
