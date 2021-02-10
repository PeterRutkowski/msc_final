import umap


class MyUMAP(umap.UMAP):
    def fit_transform(self, x):
        return super().transform(x)
    
    def fit_transform(self, x, y=None):
        return super().transform(x)
