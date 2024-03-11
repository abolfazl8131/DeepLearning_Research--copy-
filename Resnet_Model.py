import tensorflow as tf
from tensorflow.keras import layers,Model
from variables import TRAIN_FEATURES

class ResidualBlock(layers.Layer):
  
  def __init__(self, filters, **kwargs):

    super(ResidualBlock, self).__init__(**kwargs)

    self.conv1 = layers.Conv1D(filters, 3, padding="same")
    self.bn1 = layers.BatchNormalization()
    self.relu1 = layers.Activation("relu")
    self.conv2 = layers.Conv1D(filters, 3, padding="same")
    self.bn2 = layers.BatchNormalization()

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu1(x)
    x = self.conv2(x)
    x = self.bn2(x)
    return inputs + x
  


class ResNet(Model):
  

  def __init__(self, num_blocks, filters, output_dim, **kwargs):
    super(ResNet, self).__init__(**kwargs)
    self.conv1 = layers.Conv1D(filters, 7, padding="same", input_shape = (None , TRAIN_FEATURES-1))
    self.bn1 = layers.BatchNormalization()
    self.relu1 = layers.Activation("relu")

    self.blocks = tf.keras.Sequential([ResidualBlock(filters) for _ in range(num_blocks)])

    self.pool = layers.GlobalAveragePooling1D()
    self.dense = layers.Dense(output_dim, activation="softmax")

    
  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = self.relu1(x)

    x = self.blocks(x)

    x = self.pool(x)
    output = self.dense(x)
    
    return output
