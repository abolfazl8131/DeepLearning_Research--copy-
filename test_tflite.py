import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from Deeplearning.preprocess import split
from Deeplearning.variables import TRAIN_SIZE
import keras
#from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
#import matplotlib.pyplot as plt


interpreter = tf.lite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tflite/model3.tflite')
interpreter.allocate_tensors() 

output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]

input_data = tf.constant(1., shape=[1, 600,1])
interpreter.set_tensor(input['index'], input_data)
interpreter.invoke()
print(interpreter.get_tensor(output['index']).shape)
print(interpreter.get_tensor(input['index']).shape)


# model = keras.models.load_model('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tmp/checkpoint3.keras')
# tf.saved_model.save(model, "tf-model3")

# model = tf.saved_model.load('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tf-model3')
# converter = tf.lite.TFLiteConverter.from_keras_model(model) # path to the SavedModel directory
# tflite_model = converter.convert()

# # Save the model.
# with open('model3.tflite', 'wb') as f:
#   f.write(tflite_model)
