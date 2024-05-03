import tensorflow as tf
from keras.layers import (Input,Layer, 
                                     Conv1D,Conv2D, 
                                     BatchNormalization, 
                                     ReLU, 
                                     Add,
                                     Lambda,
                                     GlobalAveragePooling1D, 
                                     Dense,Activation,Flatten,Dropout,MaxPooling1D, Concatenate)
from keras.models import Model
from keras import regularizers

def conv1d(x_input,n_filter,strides,kernel_size,padding = 'valid'):

    x = Conv1D(filters=n_filter,kernel_size=kernel_size, padding=padding,strides=strides, activation='relu')(x_input)
    x = BatchNormalization()(x)
    return x

def Inception_module(x):

    conv_1 = conv1d(x_input=x,kernel_size=2,padding='same',n_filter=5,strides=1)
    conv_2 = conv1d(x_input=x,kernel_size=2,padding='same',n_filter=5,strides=1)
    conv_3 = conv1d(x_input=x,kernel_size=2,padding ='same',n_filter=5,strides=1)
    pool_1 = MaxPooling1D(pool_size=1,padding='same')(x)

    concat_x = Concatenate(axis=-1)([conv_1 , conv_2, conv_3, pool_1])

    return concat_x


def Inception_block(x):

    x = Inception_module(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x


def inception_resnet(input_shape , n_classes):
    
    x_input = Input(shape=input_shape)
    x = x_input

    x = Inception_block(x)
    

    x = Add()([x , x_input])
    x = Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    x = Flatten()(x)


    x = Dense(n_classes ,activation='softmax')(x)

    model = Model(x_input , x)

    return model