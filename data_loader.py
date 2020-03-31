import numpy as np
import sys
import pdb

sys.path.append('../recommender_system/')
from dataLoader import ClientDataLoader, DummyDataLoader

class PriceClientDataLoader(ClientDataLoader):
    def __init__(self):
        super().__init__()

class PriceDummyDataLoader(DummyDataLoader):
    def __init__(self):
        super().__init__()
        self.add_deal()

    def add_deal(self):
        new_data = np.zeros( shape=(  self.data.shape[0],self.data.shape[1]+1 ))
        new_data[:,0:self.data.shape[1]] = self.data
        new_data[:,self.data.shape[1]] = np.random.randint(0,2, self.data.shape[0])
        self.data = new_data


    def price_view(self, data):

        def add_dim(data,dim):
            return  np.hstack( (data, np.expand_dims(self.data[:,dim],axis=1) ) )

        x,_ = super().coll_view(data)
        x = np.vstack( (x[0], x[1])).T

        x=add_dim(x,4) #hoehe
        x=add_dim(x,5) #breite
        x=add_dim(x,6) #leange
        x=add_dim(x,7) #preis
        x=add_dim(x,8) #gewicht

        y = self.data[:,self.data.shape[1]-1]
        return x,y
