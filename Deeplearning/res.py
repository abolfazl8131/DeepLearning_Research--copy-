import tensorflow as tf
from keras.layers import (Input, 
                                     Conv1D,Conv2D, 
                                     BatchNormalization, 
                                     ReLU, 
                                     Add, 
                                     GlobalAveragePooling1D, 
                                     Dense,Activation,Flatten,Dropout,MaxPooling1D,GaussianNoise)
from keras.models import Model
from keras import regularizers


class Residual_block:
    kernel_size = 2
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, x_shortcut, filters):
        self.x = x
        self.filters = filters
        self.x_shortcut = x_shortcut

    def unit(self):

        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        

        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(x)
        x = BatchNormalization()(x)
        x = Activation('linear')(x)
        
       
        if x.shape[1:] == self.x_shortcut.shape[1:]:
            x = Add()([x, self.x_shortcut])
        else:
            raise Exception('Skip Connection Failure!')
        return x

class Convolution_block:
    
    kernel_size = 2
    strides = 1
    padding = 'same'
    data_format = "channels_last"

    def __init__(self, x, filters):
        self.x = x
        self.filters = filters

    def unit(self):

        x = Conv1D(self.filters, self.kernel_size, self.strides, self.padding, self.data_format)(self.x)
        x = Activation('linear')(x)
        x = BatchNormalization()(x)
        
        
        return x


def residual_stack(x, filters):

    x = Convolution_block(x, filters)
    x = x.unit() 
    x_shortcut = x

    x = Residual_block(x, x_shortcut, filters)
    x = x.unit()
    x_shortcut = x

    x = Residual_block(x, x_shortcut, filters)  
    x = x.unit()
    
    
    x = MaxPooling1D(pool_size=2,padding='same')(x)
    
    return x



def ResNet(input_shape, classes):   
    
    x_input = Input(input_shape)
    x = x_input
    
    num_filters = 5


    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
    x = residual_stack(x, num_filters)
   
    
    x = Flatten()(x)
    

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)

    x = Dense(128,activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(.5)(x)

    #x = GaussianNoise(0.5)(x) 
    x = Dense(classes , activation='softmax')(x)
    
    model = Model(inputs = x_input, outputs = x)
     
    return model