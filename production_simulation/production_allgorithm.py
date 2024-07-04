import tensorflow as tf
import numpy as np
from queue import Queue

q = Queue(maxsize = 1)

def _normalize(slice:np.ndarray) -> np.ndarray:
    #shape = (150,4)
    flatten = slice.flatten()
    min_val = np.min(flatten)
    max_val = np.max(flatten)
    scaled_data = (flatten - min_val) / (max_val - min_val)
    return scaled_data

kv_dict = {
    1:'status1',
    2:'status2',
    3:'status3',
    4:'status4'
}

def predict(X:np.ndarray):


    interpreter = tf.lite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tflite/model3.tflite')
    interpreter.allocate_tensors() 
    my_signature = interpreter.get_signature_runner()

    x = X.reshape([1,600,1])
    #print(x.shape)
    x = tf.cast(x, tf.float32)
    output = my_signature(inputs=x)

    pred_label = tf.argmax(output['output_0'],axis=1)
    print("predicted:",kv_dict[np.array(pred_label)[0]])


X = []
while True:
    sampl = np.random.uniform(low=4.5, high=6.8, size=(4,))
    X.append(sampl)
    if len(X) == 150:
        X_ = np.array(X)
        X.clear()
        X_ = _normalize(X_)
        predict(X_)

    

