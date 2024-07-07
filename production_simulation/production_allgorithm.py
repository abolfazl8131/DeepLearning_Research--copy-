import tflite_runtime.interpreter as tflite
import numpy as np
from queue import Queue
import threading


# flatten ans normalize the input data of 150 samples of 4 size array (600)
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

#we have the tflite model and give it data for prediction
def predict(X:np.ndarray):


    interpreter = tflite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tflite/model3.tflite')
    interpreter.allocate_tensors() 
    my_signature = interpreter.get_signature_runner()
    #set the shape for tflite accepted input
    x = X.reshape([1,600,1])

    #change the dtype of float32
    x = x.astype(np.float32)

    output = my_signature(inputs=x)

    pred_label = np.argmax(output['output_0'],axis=1)
    #giving prediction with dict
    return kv_dict[np.array(pred_label)[0]]


def pipeline(X:np.ndarray):
    X_ = _normalize(X)
    return predict(X_)


#######################################################################simulation########################################################

#the function generate the array of size 4 (num of sensors) with given range
def generate():
    sampl = np.random.uniform(low=3.2, high=100.0, size=(4,))
    return sampl


def simulate():
    #we have queue of size 150 (for each cycle)
    X = Queue(maxsize=150)

    #implement the enqueue and dequeue and give data to pipeline
    while True:
        
        sampl = generate()
        
        if X.full():
            X_ = np.array(list(X.queue))
            print(pipeline(X_))
            X.get()

        X.put(sampl)    
        
            
        


if __name__ =="__main__":
    #using 2 threads for data generator and the prediction_simulator
    t1 = threading.Thread(target=simulate, args=())
    t2 = threading.Thread(target=generate, args=())
    t1.start()
    t2.start()
    print("Done!")

   

