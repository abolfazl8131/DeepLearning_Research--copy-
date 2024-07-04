import pandas as pd 
import numpy as np
from keras.utils import to_categorical
import glob
import os
import tensorflow as tf


def _normalize(slice:np.ndarray) -> np.ndarray:
    #shape = (150,4)
    flatten = slice.flatten()
    min_val = np.min(flatten)
    max_val = np.max(flatten)
    scaled_data = (flatten - min_val) / (max_val - min_val)
    return scaled_data

    

    

def prepare():
    
    path = 'classification-of-defects-in-a-rotary-machine/train'
    csv_files = glob.glob(os.path.join(path, "*.csv")) 
    
    data = []
    
    
    for f in csv_files: 
                
            label = f.split('/')[2].split('.')[0][-1]
            df = pd.read_csv(f)
            
            a = np.array_split(df.iloc[:,0:4], 4000)
            

            for j in range(0,4000):
                a[j] = _normalize(np.asarray(a[j]))
                
                data.append(
                        
                    [np.array(a[j][:600], dtype=np.float64),
                    np.int32(label)]

                )
       
    
    df = pd.DataFrame(data, columns=['TS','label'])

    df = df.sample(frac = 1)

    print(len(df['TS']))

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