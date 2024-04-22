import tensorflow as tf
from keras.layers import (Input, 
                        LSTM,     
                        Dense,)
from keras.models import Model,Sequential
from keras import regularizers

def lstm_clf(input_shape:tuple, classes:int):
    x_input = Input(input_shape)
    x = x_input

    x = LSTM(4, input_shape=input_shape)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs = x_input, outputs = x)
    
    return model
