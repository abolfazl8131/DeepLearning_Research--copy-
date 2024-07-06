import tflite_runtime.interpreter as tflite
import numpy as np
from queue import Queue
import threading



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


    interpreter = tflite.Interpreter('/home/abolfazl/Desktop/DeepLearning_Research (copy)/tflite/model3.tflite')
    interpreter.allocate_tensors() 
    my_signature = interpreter.get_signature_runner()

    x = X.reshape([1,600,1])
    
    x = x.astype(np.float32)
    output = my_signature(inputs=x)

    pred_label = np.argmax(output['output_0'],axis=1)
    return kv_dict[np.array(pred_label)[0]]


def pipeline(X:np.ndarray):
    X_ = _normalize(X)
    return predict(X_)

def generate():
    sampl = np.random.uniform(low=3.2, high=100.0, size=(4,))
    return sampl

def simulate():
    X = Queue(maxsize=150)

    while True:
        
        sampl = generate()
        
        if X.full():
            X_ = np.array(list(X.queue))
            print(pipeline(X_))
            X.get()

        X.put(sampl)    
        
            
        


if __name__ =="__main__":
    t1 = threading.Thread(target=simulate, args=())
    t2 = threading.Thread(target=generate, args=())

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    print("Done!")

