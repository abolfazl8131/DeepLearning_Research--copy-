import pandas as pd 
import numpy as np


#onehot encode

def get_dummies():
    data = pd.read_csv('sample.csv')

    data['lable'] = pd.Categorical(

        data['lable'],

        categories=[1,2,3,4]
    )

    dummies = pd.get_dummies(data['lable'], dtype = int)

    data = pd.concat([data,dummies], axis=1)

    data = data.drop(['lable','Unnamed: 0'], axis=1)

    return data



def split(TRAIN_SIZE, TRAIN_FEATURES,OUTPUT_DIM):

    data = get_dummies()
    
    X_train = np.array(data.iloc[:TRAIN_SIZE , :TRAIN_FEATURES]).reshape(TRAIN_SIZE,1,TRAIN_FEATURES)

    X_test = np.array(data.iloc[TRAIN_SIZE: , :TRAIN_FEATURES]).reshape(data.shape[0]-TRAIN_SIZE,1,TRAIN_FEATURES)

    Y_train = np.array(data.iloc[:TRAIN_SIZE, TRAIN_FEATURES:]).reshape(TRAIN_SIZE,OUTPUT_DIM)

    Y_test = np.array(data.iloc[TRAIN_SIZE: , TRAIN_FEATURES:]).reshape(data.shape[0]-TRAIN_SIZE,OUTPUT_DIM)


    return X_train,X_test,Y_train,Y_test