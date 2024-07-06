
from concurrent import futures
import logging
from collections import deque
import grpc
import signal_pb2
import signal_pb2_grpc
from queue import Queue
import threading
from production_allgorithm import pipeline


import numpy as np

class Signal(signal_pb2_grpc.SignalServicer):

    signal_que = Queue(maxsize = 150)
    
    def SendSignal(self, request, context):

        
        self.signal_que.put(request.signal)
        
        #print(self.signal_array.qsize())
        try:
            li = list(self.signal_que.queue)
            np_li = np.array(li)
            
            pred = pipeline(np_li)
            return signal_pb2.Prediction(predicton=pred)
        
        except Exception as e:

            return signal_pb2.Prediction(predicton='error')
        

def serve():
    port = "50055"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    signal_pb2_grpc.add_SignalServicer_to_server(Signal(), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig()
    serve()
    