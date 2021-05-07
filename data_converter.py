import data_split
from datetime import datetime
from PIL import Image
import numpy as np
import salt_pepper_noise
import gaussian_noise
from skimage.filters import gaussian
import os
import shutil
import multiprocessing as mp


def create_split(path='in10', save_path='data/in10_split', test_size=3/13, random_state=69):
    ds = data_split.DataSplit()
    ds.save_split(path, save_path, test_size, random_state)


def perturb_img(input_data):

    index = input_data[0]
    path = input_data[1]
    blur = input_data[2]
    mode = input_data[3]
    save_path = input_data[4]
    model = input_data[5]
    img = np.asarray(Image.open(path))

    if blur == 'salt_pepper_noise':
        img = salt_pepper_noise.SaltPepperNoise().noise(img, mode)
    elif blur == 'gaussian_noise':
        img = gaussian_noise.GaussianNoise().noise(img, mode)
    elif blur == 'gaussian_blur':
        img = gaussian(img, sigma=mode, preserve_range=True)

    img = img.astype('uint8')
    perturbed = model.predict(Image.fromarray(img, 'RGB'))

    np.savez_compressed('{}/perturbs/{}_{}_{}'.format(save_path, blur, mode, index),
                        data=perturbed)


def perturb(model, paths, set_type, save_path, blur, mode):
    start_time = datetime.now()
    print(set_type, blur, mode, end=' ')
    if not os.path.isfile('{}/{}_{}_{}.npz'.format(save_path, set_type, blur, mode)):
        try:
            os.mkdir('{}/perturbs'.format(save_path))
        except FileExistsError:
            shutil.rmtree('{}/perturbs'.format(save_path))
            os.mkdir('{}/perturbs'.format(save_path))

        to_perturb = [[i, path, blur, mode, save_path, model]
                      for i, path in enumerate(paths)]

        pool = mp.Pool(70 if int(mp.cpu_count()) > 70 else int(0.8*mp.cpu_count()))
        pool.map(perturb_img, to_perturb)

        features = list()
        for index, _ in enumerate(paths):
            with np.load('{}/perturbs/{}_{}_{}.npz'.format(save_path, blur, mode, index),
                         allow_pickle=True) as f:
                features.append(f['data'])

        np.savez_compressed('{}/{}_{}_{}.npz'.format(save_path, set_type, blur, mode),
                            data=np.asarray(features))
        shutil.rmtree('{}/perturbs'.format(save_path))
    print(datetime.now() - start_time)
