import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from Deeplearning.preprocess import split
from Deeplearning.variables import TRAIN_SIZE
import keras

_,_,X,Y = split(TRAIN_SIZE)


interpreter = tf.lite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tflite/model3.tflite')
interpreter.allocate_tensors() 
my_signature = interpreter.get_signature_runner()
x = X[0].reshape([1,600,1])
print(x.shape)
x = tf.cast(x, tf.float32)
output = my_signature(inputs=x)

pred_label = tf.argmax(output['output_0'],axis=1)
print("predicted:",pred_label)
print("real",tf.argmax(Y[0]))

# model = keras.models.load_model('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tmp/checkpoint3.keras')
# tf.saved_model.save(model, "tf-model3")

# model = tf.saved_model.load('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tf-model3')
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
# tflite_model = converter.convert()

# # Save the model.
# with open('model3.tflite', 'wb') as f:
#   f.write(tflite_model)
