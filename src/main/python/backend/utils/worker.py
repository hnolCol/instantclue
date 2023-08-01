
from PyQt5.QtCore import QRunnable, pyqtSlot
from .workerSignal import WorkerSignals

import traceback
import sys

class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and 
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, funcKey, ID, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.ID = ID
        self.funcKey = funcKey
        self.signals = WorkerSignals()   

        # Add the callback to our kwargs
        #self.kwargs['progress_callback'] = self.signals.progress     

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        # Retrieve args/kwargs here; and fire processing using them
        result = {}
       # self.kwargs["signals"] = self.signals
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit({"funcKey":self.funcKey,"data":result})
              # Return the result of the processing
        finally:
            #check if appropiate details are present in results (e.g. to show a message)
            msg  = " "
            if isinstance(result,dict) and "messageProps" in result:
                msg = f"{result['messageProps']['title']}: {result['messageProps']['message']}"
            self.signals.finished.emit(self.ID, msg)  # Done
