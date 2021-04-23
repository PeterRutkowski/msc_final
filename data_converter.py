import data_split
from tqdm import tqdm
from PIL import Image
import numpy as np
import salt_pepper_noise
import gaussian_noise
from skimage.filters import gaussian


class DataConverter:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def create_split(path='data', save_path='data/in10_split', test_size=3/13, random_state=69):
        ds = data_split.DataSplit()
        ds.save_split(path, save_path, test_size, random_state)
    
    def perturb(self, paths, set_type, save_path, blur, mode):
        images = list()
        predictions = list()
        for i in tqdm(range(len(paths)), desc='[{} {}]'.format(blur, mode)):
            img = np.asarray(Image.open(paths[i]))
            if blur == 'salt_pepper_noise':
                img = salt_pepper_noise.SaltPepperNoise().noise(img, mode)
            elif blur == 'gaussian_noise':
                img = gaussian_noise.GaussianNoise().noise(img, mode)
            elif blur == 'gaussian_blur':
                img = gaussian(img, sigma=mode, preserve_range=True)

            img = img.astype('uint8')

            images.append(img)
            predictions.append(self.model.predict(Image.fromarray(img, 'RGB')))

        np.savez_compressed('{}/images_{}_{}_{}'.format(save_path, set_type, blur, mode),
                            data=np.asarray(images, dtype=object))
        np.savez_compressed('{}/{}_{}_{}'.format(save_path, set_type, blur, mode),
                            data=np.asarray(predictions))
