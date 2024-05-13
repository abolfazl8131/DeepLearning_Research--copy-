import keras

checkpoint_filepath = './tmp/checkpoint3.keras'


model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


tensorboard = keras.callbacks.TensorBoard(
    log_dir='./tensorboard'
)

early_stopping = keras.callbacks.EarlyStopping(monitor = "val_loss")