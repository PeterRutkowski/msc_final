import numpy as np
from sklearn.svm import SVC
import multiprocessing as mp
import joblib
from os import walk


def component_model(train_data):
    x, y = train_data[0], train_data[1]
    clf = SVC(kernel='poly', degree=2)
    clf.fit(x, y)
 
    joblib.dump(clf, 'pipeline_data/comp120_dbscan200/model_{}'.format(train_data[2]),
                compress='lzma')


x_train = np.load('pipeline_data/x_train_none_none.npz', allow_pickle=True)['data']
y_train = np.load('pipeline_data/comp120_dbscan200/rep_x_train.npz', allow_pickle=True)['data']

_, _, filenames = next(walk('pipeline_data/comp120_dbscan200'))

done = list()
for filename in filenames:
    if filename.startswith('model'):
        done.append(int(filename[6:]))

todo = [n for n in range(y_train.shape[1])]

for n in done:
    todo.remove(n)

print(todo)
pool = mp.Pool(int(mp.cpu_count()/2))
pool.map(component_model, [[x_train, y_train[:, i], i] for i in todo])
