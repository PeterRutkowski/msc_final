import numpy as np
from tqdm import tqdm


class Binarizer:
    def __init__(self):
        pass

    @staticmethod
    def binarize(x, graphs):
        """Generates a binary representation of x based on Mapper
        graphs.

        :param x: Images.
        :type x: numpy.ndarray

        :return: Binary representation of x.
        :rtype: numpy.ndarray
        """

        rep = []

        for i in tqdm(range(len(x)), desc='[binarization]'):
            row_rep = []
            for comp in range(len(graphs)):
                graph = graphs[comp]
                graph_rep = [0 for _ in range(graph.vcount())]
                for j, node in enumerate(graph.vs):
                    graph_rep[j] = 1 if i in node['node_elements'] else 0
                row_rep += graph_rep
            rep.append(row_rep)

        return np.asarray(rep)
