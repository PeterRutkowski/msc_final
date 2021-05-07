import data_split
from tqdm import tqdm
from PIL import Image
import numpy as np
import salt_pepper_noise
import gaussian_noise
from skimage.filters import gaussian
import os
import shutil
import multiprocessing as mp

class DataConverter:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def create_split(path='in10', save_path='data/in10_split', test_size=3/13, random_state=69):
        ds = data_split.DataSplit()
        ds.save_split(path, save_path, test_size, random_state)

    @staticmethod
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
    
    def perturb(self, paths, set_type, save_path, blur, mode):
        predictions = list()
        if not os.path.isfile('{}/{}_{}_{}.npz'.format(save_path, set_type, blur, mode)):
            try:
                os.mkdir('{}/perturbs'.format(save_path))
            except FileExistsError:
                shutil.rmtree('{}/perturbs'.format(save_path))
                os.mkdir('{}/perturbs'.format(save_path))

            to_perturb = [[i, path, blur, mode, save_path, self.model]
                          for i, path in enumerate(paths)]

            pool = mp.Pool(70 if int(mp.cpu_count()) > 70 else mp.cpu_count())
            pool.map(self.perturb_img, to_perturb)



        for i in tqdm(range(len(paths)), desc='[{} {}]'.format(blur, mode)):
            img = np.asarray(Image.open(paths[i]))
            if blur == 'salt_pepper_noise':
                img = salt_pepper_noise.SaltPepperNoise().noise(img, mode)
            elif blur == 'gaussian_noise':
                img = gaussian_noise.GaussianNoise().noise(img, mode)
            elif blur == 'gaussian_blur':
                img = gaussian(img, sigma=mode, preserve_range=True)

            img = img.astype('uint8')

            predictions.append(self.model.predict(Image.fromarray(img, 'RGB')))

        np.savez_compressed('{}/{}_{}_{}'.format(save_path, set_type, blur, mode),
                            data=np.asarray(predictions))
