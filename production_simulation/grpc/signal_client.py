from __future__ import print_function

import logging
import time
import grpc
import signal_pb2
import signal_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    while True:
        with grpc.insecure_channel("localhost:50055") as channel:
            stub = signal_pb2_grpc.SignalStub(channel)
            response = stub.SendSignal(signal_pb2.SignalREQ(signal=[2.0,2.1,3.1,2.5]))
            print(response.predicton)
            #time.sleep(1)
            #print("Engine status: " + response.predicton)


if __name__ == "__main__":
    logging.basicConfig()
    run()