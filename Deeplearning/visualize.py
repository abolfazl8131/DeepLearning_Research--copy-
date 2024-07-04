import keras

from res import ResNet as res

from ispl import ispl_inception
import keras

from variables import *

from res import ResNet as res

from keras.utils.vis_utils import plot_model

plot_model(res(), to_file='ResNet.png', show_shapes=True, show_layer_names=True)


