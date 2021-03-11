from sklearn.decomposition import PCA


class MyPCA(PCA):
    def fit_transform(self, x):
        return super().transform(x)
    
    def fit_transform(self, x, y=None):
        return super().transform(x)
