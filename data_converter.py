import cv2
import data_split
from tqdm import tqdm
from PIL import Image
import numpy as np


class DataConverter:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def create_split(path='data', save_path='data/in10_split', test_size=3/13, random_state=69):
        ds = data_split.DataSplit()
        ds.save_split(path, save_path, test_size, random_state)
    
    def convert(self, paths, noise='none'):
        images = []
        
        for i in tqdm(range(len(paths)), desc='[Converting images]'.format(noise)):
            img = np.asarray(Image.open(paths[i]))
            if noise == 'gaussian':
                img = cv2.GaussianBlur(img, (5, 5), 0)
            images.append(self.model.predict(Image.fromarray(img.astype('uint8'), 'RGB')))
        
        return np.asarray(images)
    
    def convert_split(self, path='data/in10_split.npz', save_path='data/in10_split_converted'):
        loaded = np.load(path)
        
        np.savez(save_path,
                 x_test_none=self.convert(loaded['paths_test']),
                 #x_test_gaussian=self.convert(loaded['paths_test'], 'gaussian'),
                 x_train=self.convert(loaded['paths_train']),
                 y_train=loaded['labels_train'],
                 y_test=loaded['labels_test'])
