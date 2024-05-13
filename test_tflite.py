import tensorflow as tf
import numpy as np
import tensorflow as tf
from Deeplearning.preprocess import split
from Deeplearning.variables import TRAIN_SIZE
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


interpreter = tf.lite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/ispl.tflite')
interpreter.allocate_tensors() 

output = interpreter.get_output_details()[0]
input = interpreter.get_input_details()[0]

input_data = tf.constant(1., shape=[1, 600,1])
interpreter.set_tensor(input['index'], input_data)
interpreter.invoke()
print(interpreter.get_tensor(output['index']).shape)
print(interpreter.get_tensor(input['index']).shape)


# X_train,Y_train,X_Test,Y_Test = split(TRAIN_SIZE)
# y_pred = np.argmax(model.predict(X_Test), axis=1)
# Y_Test = np.argmax(Y_Test, axis=1)
# result = confusion_matrix(Y_Test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=result)
# disp.plot()
# plt.savefig('conf_matrix.png')