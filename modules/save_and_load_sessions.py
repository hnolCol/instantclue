"""
	""SAVE AND LOAD SESSIONS""
	* Data are presented by their column names
	* This classs handles clicks by user, filters unwanted selection
	* and organizes the data.
	
    Instant Clue - Interactive Data Visualization and Analysis.
    Copyright (C) Hendrik Nolte

    This program is free software; you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation; either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
"""

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
	
   	
            	
	
             
	
	
	
	
	
	





