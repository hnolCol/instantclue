import os
import tkinter as tk
import tkinter.filedialog as tf
import tkinter.simpledialog as ts
import pickle
import time

from modules.utils import *


def save_session(classDict):
	'''
	'''
	sessionName = ts.askstring('Provide session name',
								prompt = 'Session name: ',
								initialvalue= time.strftime("%d_%m_%Y"))
	if sessionName is None:
		return False						
	sessionPath = os.path.join(path_file,'Data','stored_sessions',sessionName)
	
	while os.path.exists(sessionPath):
		
		quest = tk.messagebox.askquestion('Path exists already...',
										  'Session folder exists already. Overwrite content?')
		if quest == 'yes':
			break
		else:
			sessionName = ts.askstring('Provide session name',
								prompt = 'Session name: ',
								initialvalue= time.strftime("%d_%m_%Y"))
			sessionPath = os.path.join(path_file,'Data','stored_sessions',sessionName)
	
	if sessionPath is None:
		return
	
	if not os.path.exists(sessionPath):
		try:
			os.makedirs(sessionPath)	
		except:
			tk.messagebox.showinfo('Make dir failed ..','Creation of dir failed. Permission?')
			return	
		
	
	with open(os.path.join(sessionPath,'{}.pkl'.format(sessionName)),'wb') as file:
		pickle.dump(classDict, file)
	
	
def open_session():
	'''
	'''
	directorySession = tf.askdirectory(initialdir = os.path.join(path_file,'Data','stored_sessions'), title ="Choose saved session")	
	if directorySession is None or directorySession == '':
		return
		
	dictPath = None
	for root, dirs, files in os.walk(directorySession):
		for file in files:
			if file.endswith(".pkl"):
				dictPath = os.path.join(root, file)
				break	
	if dictPath is None:
		return 'Not pckl found'
	with open(dictPath,'rb') as file:
		classDict = pickle.load(file)
		
	return classDict
	
   	
            	
	
             
	
	
	
	
	
	





