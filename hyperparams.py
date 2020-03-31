from keras.optimizers import Adam, Nadam, RMSprop
import random
from keras.losses import binary_crossentropy
from keras.activations import relu,elu
from keras.activations import sigmoid

def get_random_hp():
    hp = {
            'lr': (0.8, 1.2, 3),
            'first_neuron': [4, 8, 16, 32, 64],
            'hidden_layers': [0, 1, 2],
            'batch_size': (2,4,6,12,18,24,32,64),
            #'epochs': [1,2,3,4,5,6,7,8,9,10,50, 100, 150],
            'epochs': [1,2,3,4,5,6,7,8,9,10,50],
            'dropout': (0, 0.2, 3),
            'weight_regulizer': [None],
            'emb_output_dims': [None],
            'shape': ['brick', 'long_funnel'],
            'kernel_initializer': ['uniform', 'normal'],
            'optimizer': [Adam, Nadam, RMSprop],
            'losses': [binary_crossentropy],
            'activation': [relu, elu],
            'last_activation': [sigmoid]
    }

    for key in hp:
        hp[key] = random.choice(hp[key])

    return hp