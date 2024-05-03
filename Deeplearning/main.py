import keras
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import tensorflow as tf
from preprocess import split
from variables import *
import numpy as np
from res import ResNet as res
import matplotlib.pyplot as plt
import random
from contextlib import redirect_stdout
from callbacks import model_checkpoint_callback,tensorboard
from optimizers import adam
from simple import simple_conv

model = simple_conv(INPUT_SHAPE, OUTPUT_DIM)

X_train,Y_train,X_Test,Y_Test = split(TRAIN_SIZE)


model.compile(optimizer=adam, 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])


#monitoring resnet archirecture
with open('summery.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


history = model.fit(
        X_train,
        Y_train, 
        epochs=EPOCHES,
        batch_size = 128,
        validation_data=[X_Test, Y_Test],
        callbacks=[model_checkpoint_callback,tensorboard]
         
)


#conf matrix
y_pred = np.argmax(model.predict(X_Test), axis=1)
Y_Test = np.argmax(Y_Test, axis=1)
result = confusion_matrix(Y_Test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=result)
disp.plot()
plt.savefig('conf_matrix.png')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_accuracy.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('model_loss.png')




