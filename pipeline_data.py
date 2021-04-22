import data_converter
import vgg19bn
import numpy as np

dc = data_converter.DataConverter(vgg19bn.VGG19bn(layers=[60]))

dc.create_split(path='data', save_path='pipeline_data/in10_split')

data_paths = np.load('pipeline_data/in10_split.npz')

# labels

np.savez_compressed('pipeline_data/y_train', data=data_paths['labels_train'])
np.savez_compressed('pipeline_data/y_test', data=data_paths['labels_test'])

# training set

dc.perturb(paths=data_paths['paths_train'],
           set_type='x_train',
           save_path='pipeline_data',
           blur='none',
           mode='none')

# testing set

dc.perturb(paths=data_paths['paths_test'],
           set_type='x_test',
           save_path='pipeline_data',
           blur='none',
           mode='none')

for thr in np.arange(0.03, 0.36, 0.03):
    dc.perturb(paths=data_paths['paths_test'],
               set_type='x_test',
               save_path='pipeline_data',
               blur='salt_pepper_noise',
               mode=thr)

for sd in np.arange(10, 120, 10):
    dc.perturb(paths=data_paths['paths_test'],
               set_type='x_test',
               save_path='pipeline_data',
               blur='gaussian_noise',
               mode=sd)

for sigma in np.arange(0.5, 6, 0.5):
    dc.perturb(paths=data_paths['paths_test'],
               set_type='x_test',
               save_path='pipeline_data',
               blur='gaussian_blur',
               mode=sigma)
