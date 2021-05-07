import data_split
from datetime import datetime
from PIL import Image
import numpy as np
import salt_pepper_noise
import gaussian_noise
from skimage.filters import gaussian
import os
import shutil
from tqdm import tqdm
import vgg19bn


def create_split(path, save_path, test_size=3/13, random_state=69):
    if not os.path.isfile('{}/in10_split.npz'.format(save_path)):
        ds = data_split.DataSplit()
        ds.save_split(path, save_path, test_size, random_state)


def perturb_img(index, path, blur, mode, save_path, vgg):
    img = np.asarray(Image.open(path))

    if blur == 'salt_pepper_noise':
        img = salt_pepper_noise.SaltPepperNoise().noise(img, mode)
    elif blur == 'gaussian_noise':
        img = gaussian_noise.GaussianNoise().noise(img, mode)
    elif blur == 'gaussian_blur':
        img = gaussian(img, sigma=mode, preserve_range=True)

    img = Image.fromarray(img.astype('uint8'), 'RGB')

    np.savez_compressed('{}/perturbs/{}_{}_{}'.format(save_path, blur, mode, index),
                        data=vgg.predict(img))


def perturb(paths, set_type, save_path, blur, mode):
    start_time = datetime.now()
    print(set_type, blur, mode, end=' ')
    if not os.path.isfile('{}/{}_{}_{}.npz'.format(save_path, set_type, blur, mode)):
        try:
            os.mkdir('{}/perturbs'.format(save_path))
        except FileExistsError:
            shutil.rmtree('{}/perturbs'.format(save_path))
            os.mkdir('{}/perturbs'.format(save_path))

        vgg = vgg19bn.VGG19bn(layers=[53])

        for index, path in enumerate(tqdm(paths)):
            perturb_img(index, path, blur, mode, save_path, vgg)

        features = list()
        for index, _ in enumerate(paths):
            with np.load('{}/perturbs/{}_{}_{}.npz'.format(save_path, blur, mode, index),
                         allow_pickle=True) as f:
                features.append(f['data'])

        np.savez_compressed('{}/{}_{}_{}.npz'.format(save_path, set_type, blur, mode),
                            data=np.asarray(features))
        shutil.rmtree('{}/perturbs'.format(save_path))
    print(datetime.now() - start_time)
