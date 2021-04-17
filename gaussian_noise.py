import numpy as np


class GaussianNoise:
    def __init__(self):
        pass

    @staticmethod
    def noise(img, sd):
        new_img = np.copy(img)
        for i in range(new_img.shape[0]):
            for j in range(new_img.shape[1]):
                new_pixel = new_img[i, j, :] + np.random.normal(0, sd, 3)
                for k in range(new_pixel.shape[0]):
                    if new_pixel[k] < 0:
                        new_pixel[k] = 0
                    elif new_pixel[k] > 255:
                        new_pixel[k] = 255
                new_img[i, j, :] = new_pixel

        return new_img
