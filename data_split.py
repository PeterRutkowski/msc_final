import os
import numpy as np
from sklearn.model_selection import train_test_split


class DataSplit:
    def __init__(self):
        self.label_dict = {'african_elephant': 0,
                           'agaric': 1,
                           'eastern_grey_squirrel': 2,
                           'hawk': 3,
                           'koala': 4,
                           'lichen': 5,
                           'meerkat': 6,
                           'phlox': 7,
                           'tamandua': 8,
                           'wasp_s_nest': 9}
    
    @staticmethod
    def get_paths_from_folder(folder_path):
        root, _, filenames = next(os.walk(folder_path))
        
        return [os.path.join(root, filename) for filename in filenames]
        
    def save_split(self, path, save_path='in10/in10_split', test_size=3/13, random_state=69):
        root, foldernames, filenames = next(os.walk(path))
        paths, labels = [], []
        for foldername in foldernames:
            paths += self.get_paths_from_folder(os.path.join(root, foldername))
            labels += [self.label_dict[foldername] for _ in range(1300)]
            
        p_train, p_test, l_train, l_test = train_test_split(np.asarray(paths), 
                                                            np.asarray(labels), 
                                                            test_size=test_size, 
                                                            shuffle=True, 
                                                            random_state=random_state, 
                                                            stratify=labels)
        
        np.savez(os.path.join(save_path),
                 paths_train=p_train,
                 paths_test=p_test, 
                 labels_train=l_train, 
                 labels_test=l_test) 
