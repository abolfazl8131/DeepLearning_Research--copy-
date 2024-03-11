from Resnet_Model import ResNet
from preprocess import split
from variables import *

model = ResNet(filters=FILTERS , num_blocks=N_BLOCKS , output_dim=OUTPUT_DIM)
X_train,X_test,Y_train,Y_test = split(TRAIN_SIZE, TRAIN_FEATURES,OUTPUT_DIM)
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train,Y_train, epochs=EPOCHES)