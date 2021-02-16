
import time 
import subprocess
import os 
import sys
import pickle
import numpy as np 

control_data=bytes([1])
control_stop=bytes([0])



class SubProcessCall(object):

    def __init__(self, cmd, arr):
        ""
        self.cmd = cmd
        self.arr = arr
        print(self.cmd)
        self.initSubProcess()

    def initSubProcess(self):
        ""
        self.subProcess = subprocess.Popen(self.cmd, stdout=subprocess.PIPE,  stdin=subprocess.PIPE)

    def run(self):
        ""
        print(type(self.arr))
        if isinstance(self.arr,np.ndarray):
            self.sendArray(self.arr)
            self.sendStop()
            for stdoutLine in self.subProcess.stdout:
                path = stdoutLine.decode("utf-8")
                if os.path.exists(path):
                    responseArray = np.load(path)
                    os.remove(path)
            
            return responseArray
            
        else:
            raise ValueError("arr must be a numpy array.")
        
    
    def sendArray(self, arr):
        ""
        dataStr=pickle.dumps(arr)  #pickle the data array into a byte array
        dlen=len(dataStr).to_bytes(8, byteorder='big') #find the length of the array and
        self.sendControl()
        self.subProcess.stdin.write(dlen)
        self.subProcess.stdin.write(dataStr)
        self.subProcess.stdin.flush() #not sure this needed
  
    def sendControl(self):
        ""
        self.subProcess.stdin.write(control_data)

    def sendStop(self):
        ""
        self.subProcess.stdin.write(control_stop)
        self.subProcess.stdin.flush()




