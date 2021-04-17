import numpy as np


class SaltPepperNoise:
    def __init__(self):
        pass

    @staticmethod
    def noise(img, thr):
        new_img = np.copy(img)
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                sample = np.random.uniform()
                if sample < thr / 2:
                    new_img[i, j, :] = 0
                elif sample > 1 - (thr / 2):
                    new_img[i, j, :] = 255

        return new_img
