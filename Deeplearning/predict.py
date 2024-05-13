from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
from preprocess import split
from variables import TRAIN_SIZE
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

model = load_model('./tmp/checkpoint.keras', custom_objects=None, compile=True, safe_mode=True)
X_train,Y_train,X_Test,Y_Test = split(TRAIN_SIZE)
y_pred = np.argmax(model.predict(X_Test), axis=1)
Y_Test = np.argmax(Y_Test, axis=1)
result = confusion_matrix(Y_Test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=result)
disp.plot()
plt.savefig('conf_matrix.png')

