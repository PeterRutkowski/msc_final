import numpy as np
import hidden_prints
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class WeightedKNN:
    def __init__(self):
        pass

    @staticmethod
    def get_component_alphas(proj, cover, delta=0.1):
        """Assigns image's latent space projection to cover's intervals.
        Projection is assigned to an interval if
        dist(proj, interval_midpoint) < interval_length*(1+delta)/2.

        :param proj: Projection of an image onto latent space's single
        component.
        :type proj: float
        :param cover: Cover of latent space's single component.
        :type cover: list
        :param delta: Parameter scaling intervals' lengths.
        :type delta: float

        :return: Indices of cover intervals onto which image was
        projected.
        :rtype: tuple
        """

        alphas = []
        if proj < cover[0][1]:
            alphas.append(0)
        for i in range(1, len(cover)-1):
            midpoint = (cover[i][0] + cover[i][1])/2
            if abs(proj - midpoint) < abs(cover[i][1] - cover[i][0])*(1+delta)/2:
                alphas.append(i)
        if proj > cover[-1][0]:
            alphas.append(len(cover)-1)

        return tuple(alphas)
    
    def get_alphas(self, x, covers, latent_space):
        """Assigns all images' latent space projections to covers'
        intervals.

        :param x: Images.
        :type x: numpy.ndarray
        :param covers: Covers of latent space's components.
        :type covers: list
        :param latent_space: Latent space projector.
        :type latent_space: latent_space.LatentSpace

        :return: Interval assignments of images' projections.
        :rtype: list
        """
        
        with hidden_prints.HiddenPrints():
            x_latent = latent_space.transform(x)
        
        return [[self.get_component_alphas(img_latent[i], covers[i])
                 for i in range(len(img_latent))] for img_latent in x_latent]

    @staticmethod
    def get_cover_preimages(x, covers, latent_space, n=0):
        """Calculates cover preimages of latent space's nth component.

        :param x: Images.
        :type x: numpy.ndarray
        :param covers: Covers of latent space's components.
        :type covers: list
        :param latent_space: Latent space projector.
        :type latent_space: latent_space.LatentSpace
        :param n: Index of latent space's component.
        :type n: int

        :return preimage_ids: Ids of images split into preimages.
        :rtype preimage_ids: list
        """

        n_intervals = len(covers[0])
        x_latent = latent_space.transform(x)
        preimage_ids = []

        for i in range(n_intervals):
            interval = covers[n][i]
            interval_ids = []

            for j, projection in enumerate(x_latent[:, n]):
                if projection > interval[0]:
                    if projection < interval[1]:
                        interval_ids.append(j)
            preimage_ids.append(interval_ids)

        return preimage_ids

    @staticmethod
    def fit_knns(x, preimages, k, n=0):
        """Trains numerous KNN models on intervals' preimages.

        :param x: Images.
        :type x: numpy.ndarray
        :param preimages: Preimages of intervals covering
        latent space's nth component.
        :type preimages: list
        :param k: Number of nearest neighbours in KNN.
        :type k: int
        :param n: Index of latent space's component.
        :type n: int

        :return knns: KNNs trained on intervals' preimages.
        :rtype knns: dict
        """

        knns = {}
        n_intervals = len(preimages)

        for i in range(n_intervals):
            knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
            knn.fit(x[preimages[i]])
            knns.update({(n, tuple([i])): [knn, x[preimages[i]], np.array(preimages[i])]})
            if i > 0:
                knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
                union = list(set(preimages[i]) | set(preimages[i-1]))
                knn.fit(x[union])
                knns.update({(n, tuple([i-1, i])): [knn, x[union], np.array(union)]})
                
        return knns

    @staticmethod
    def get_nearest_neighbours(x, alphas, knns, k, n=0):
        """Calculates nearest neighbours of x's images.

        :param x: Images.
        :type x: numpy.ndarray
        :param alphas: Interval assignments of images' projections.
        :type alphas: list
        :param knns: Trained KNN models.
        :type knns: dict
        :param k: Number of nearest neighbours in KNN.
        :type k: int
        :param n: Index of latent space's component.
        :type n: int

        :return nn_ids: Ids of images' k nearest neighbours.
        :rtype nn_ids: list
        :return nn_dists: Distances of images to their k nearest
        neighbours.
        :rtype nn_dists: list
        """

        nn_ids = []
        nn_dists = []
        n_images = x.shape[0]

        for i in tqdm(range(n_images), desc='[wknn]'):
            nn_comp_ids = []
            nn_comp_dists = []

            nn = knns.get((n, alphas[i][n]))

            if len(nn[2]) > k:
                knn = nn[0].kneighbors([x[i]])
                ids = nn[2][knn[1]]
                nn_comp_ids.append(ids.squeeze())
                nn_comp_dists.append(knn[0].squeeze())
            else:
                ids = nn[2]
                dists = np.linalg.norm(x[i] - nn[1], axis=1)
                nn_comp_ids.append(ids)
                nn_comp_dists.append(dists)

            nn_ids.append(nn_comp_ids)
            nn_dists.append(nn_comp_dists)
        
        return nn_ids, nn_dists

    @staticmethod
    def get_weighted_representation(x_binary, nn_ids, nn_dists, k, mu=1e-05):
        """Calculates weighted KNN representations.

        :param x_binary: Binary representation of images on which KNNs
        were trained.
        :type x_binary: numpy.ndarray
        :param nn_ids: Ids of images' k nearest neighbours.
        :type nn_ids: list
        :param nn_dists: Distances of images to their k nearest
        neighbours.
        :type nn_dists: list
        :param k: Number of nearest neighbours in KNN.
        :type k: int
        :param mu: Parameter that ensures that there will be no
        division by 0.
        :type mu: float

        :return: Weighted KNN representation.
        :rtype: numpy.ndarray
        """

        n_features = x_binary.shape[1]
        n_images = len(nn_ids)

        weighted_rep = np.zeros((n_images, n_features))

        for i in range(n_images):
            weights = nn_dists[i][0]
            weights = 1./(weights + mu)
            weights = np.asarray(weights / sum(weights)).reshape(1, k)
            features = weights.dot(x_binary[nn_ids[i][0], :].astype(float))
            weighted_rep[i] = features

        return np.asarray(weighted_rep)

    def fit_transform(self, k, x_test, x_train, x_train_binary, latent_space, covers):
        """Generates a weighted KNN representation of x_test
        based on KNN models trained on x_train.

        :param k: Number of nearest neighbours in KNN.
        :type k: int
        :param x_test: Images without representation.
        :type x_test: numpy.ndarray
        :param x_train: Images with binary representation.
        :type x_train: numpy.ndarray
        :param x_train_binary: Binary representation of images.
        :type x_train_binary: numpy.ndarray
        :param latent_space: Latent space projector.
        :type latent_space: latent_space.LatentSpace
        :param covers: Covers of latent space's components.
        :type covers: list

        :return: Weighted KNN representation of x_test.
        :rtype: numpy.ndarray
        """
        alphas = self.get_alphas(x_test, covers, latent_space)
        
        preimages = self.get_cover_preimages(x_train, covers, latent_space)
        knns = self.fit_knns(x_train, preimages, k)
        
        nn_ids, nn_dists = self.get_nearest_neighbours(x_test, alphas, knns, k)

        return self.get_weighted_representation(x_train_binary, nn_ids, nn_dists, k)
