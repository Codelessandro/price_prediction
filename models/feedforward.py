from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

from config import config

def make_model(hp):
    model = Sequential()
    model.add(Dense(10, input_dim=config["nr_dimensions"], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



