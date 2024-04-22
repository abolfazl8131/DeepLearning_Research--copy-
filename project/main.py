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
from lstm import lstm_clf

model = res(INPUT_SHAPE, OUTPUT_DIM)


X_train,Y_train,X_Test,Y_Test = split(TRAIN_SIZE)

adam = keras.optimizers.Adam(learning_rate=0.0001)

rmsprop = keras.optimizers.RMSprop(learning_rate=0.0001)

model.compile(optimizer=adam, 
              loss='categorical_crossentropy', 
              metrics = ['accuracy'])


#monitoring resnet archirecture
with open('summery.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()


checkpoint_filepath = './tmp/checkpoint.keras'

class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_epoch_begin(self, epoch, logs=None):
    self.model.layers[-2].stddev = random.uniform(0, 1)
    print('updating sttdev in training')
    print(self.model.layers[-2],self.model.layers[-2].stddev)


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

noise_change = MyCustomCallback()


history = model.fit(
        X_train,
        Y_train, 
        epochs=EPOCHES,
        batch_size = 128,
        validation_data=[X_Test, Y_Test],
        callbacks=[model_checkpoint_callback,noise_change]
         
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

#monitoring resnet archirecture
with open('summery.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()



