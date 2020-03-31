import os
import pdb
import numpy as np

from data_loader import  PriceClientDataLoader, PriceDummyDataLoader
from train import *
from hyperparams import *

if os.environ['is_prod']=='True':
    data_loader = PriceClientDataLoader()

if os.environ['is_prod']=='False':
    data_loader = PriceDummyDataLoader()




x,y = data_loader.price_view(data_loader.data)
hp = get_random_hp()
train_feedforward(x,y,hp)


