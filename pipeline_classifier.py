from sklearn.svm import SVC
import numpy as np
import torch
import pickle
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import os


def plot(experiment_name, noise_type):
    df = pickle.load(open('pipeline_data/{}/scores'.format(experiment_name), 'rb'))
    fig = px.line(df[df['noise'].isin([noise_type, 'none'])],
                  x='noise scale',
                  y='accuracy',
                  color='model',
                  title='{} {}'.format(experiment_name, noise_type),
                  range_y=[0, 1])
    return fig


experiments = ['pca60_eps100_int4',
               'pca60_eps90_int4',
               'pca60_eps85_int4',
               'pca60_eps100_int7',
               'pca60_eps90_int7',
               'pca60_eps85_int7']

with open('pipeline_data/plots.html', 'a') as f:
    for exp_index, experiment in enumerate(experiments):
        if not os.path.isfile('pipeline_data/{}/scores'.format(experiment)):
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
                             'x_test_gaussian_blur_5.5']:
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

                print(experiment, test_set)

            pickle.dump(pd.DataFrame(scores, columns=['noise', 'model', 'noise scale', 'accuracy']),
                        open('pipeline_data/{}/scores'.format(experiment), 'wb'))

        f.write(plot(experiment, 'gaussian blur').to_html(full_html=False, include_plotlyjs='cdn'))
