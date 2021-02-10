import numpy

class LatentSpace():
    def __init__(self, projectors, label):
        self.projectors = projectors
        self.label = label
        
    def transform(self, X):
        rep = []
        for proj in self.projectors:
            rep.append(proj.transform(X))
            
        return numpy.hstack(rep)