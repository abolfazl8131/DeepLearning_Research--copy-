import pandas as pd 
import numpy as np
from keras.utils import to_categorical
import glob
import os
import tensorflow as tf



def prepare():
    
    path = 'classification-of-defects-in-a-rotary-machine/train'
    csv_files = glob.glob(os.path.join(path, "*.csv")) 
    
    data = []
    
    for f in csv_files: 
                
            label = f.split('/')[2].split('.')[0][-1]
            df = pd.read_csv(f)
            
            a = np.array_split(df.iloc[:,1], 1000)
                
            for j in range(0,1000):
                    
                data.append(
                        
                    [np.array(a[j][:600], dtype=np.float64),
                    np.int32(label)]

                )
       
            
    df = pd.DataFrame(data, columns=['TS','label'])

    df = df.sample(frac = 1)

    return df

    
def split(TRAIN_SIZE):

    df = prepare()
    
    X_Train = np.array(df['TS'].to_list())[:TRAIN_SIZE,:,tf.newaxis]

    Y_Train = to_categorical(np.array(df['label'].to_list()))[:TRAIN_SIZE]

    X_Test = np.array(df['TS'].to_list())[TRAIN_SIZE:,:,tf.newaxis]

    Y_Test = to_categorical(np.array(df['label'].to_list()))[TRAIN_SIZE:]
    
    return X_Train,Y_Train,X_Test,Y_Test


def sklearn_split(TRAIN_SIZE):
    df = prepare()
    
    X_Train = np.array(df['TS'].to_list())[:TRAIN_SIZE,:]

    Y_Train = to_categorical(np.array(df['label'].to_list()))[:TRAIN_SIZE]

    X_Test = np.array(df['TS'].to_list())[TRAIN_SIZE:,:]

    Y_Test = to_categorical(np.array(df['label'].to_list()))[TRAIN_SIZE:]
    
    return X_Train,Y_Train,X_Test,Y_Test