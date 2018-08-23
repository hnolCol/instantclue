

tsq_dtype = [
    ('size', 'int32'),  # bytes 0-4
    ('evtype', 'int32'),  # bytes 5-8
    ('evname', 'S4'),  # bytes 9-12
    ('channel', 'uint16'),  # bytes 13-14
    ('sortcode', 'uint16'),  # bytes 15-16
    ('timestamp', 'float64'),  # bytes 17-24
    ('offset', 'int64'),  # bytes 25-32
    ('dataformat', 'int32'),  # bytes 33-36
    ('frequency', 'float32'),  # bytes 37-40
]

EVTYPE_UNKNOWN = int('00000000', 16)  # 0
EVTYPE_STRON = int('00000101', 16)  # 257
EVTYPE_STROFF = int('00000102', 16)  # 258
EVTYPE_SCALAR = int('00000201', 16)  # 513
EVTYPE_STREAM = int('00008101', 16)  # 33025
EVTYPE_SNIP = int('00008201', 16)  # 33281
EVTYPE_MARK = int('00008801', 16)  # 34817
EVTYPE_HASDATA = int('00008000', 16)  # 32768
EVTYPE_UCF = int('00000010', 16)  # 16
EVTYPE_PHANTOM = int('00000020', 16)  # 32
EVTYPE_MASK = int('0000FF0F', 16)  # 65295
EVTYPE_INVALID_MASK = int('FFFF0000', 16)  # 4294901760
EVMARK_STARTBLOCK = int('0001', 16)  # 1
EVMARK_STOPBLOCK = int('0002', 16)  # 2


import tkinter as tk
from tkinter import messagebox
import numpy as np
import pandas as pd
import time
import os
from collections import OrderedDict

class TDTTankToPandas(object):
	
	
	def __init__(self,dir):
		''
		
		self.dir = dir
		
		self.check_dir(dir)
		
		if hasattr(self,'tanks'):
			
			self.read_files()
		
		
	def check_dir(self,dir):
		''
		tanknames = [os.path.basename(self.dir)]
		
		files = os.listdir(self.dir)
		files = self.get_tdt_files(files)
		
		self.tanks = self.check_files(files,tanknames = tanknames)
		
		if self.tanks is None:
			return
		
		
		
		#self.extract_file_names(files)
		
		
		#print(self.tanknames)
		#print(files)




	def check_files(self,files,endings = ['Tbk','tev','Tdx','tsq'], tanknames = []):
		''		
		tanks = OrderedDict()
		for ending in endings:
			nTdtFiles = len([file for file in files if file.split('.')[-1] == str(ending)])
			if nTdtFiles != 1:
				
				if nTdtFiles > 1:
					
					quest = messagebox.askquestion('Error..','Found more than one file of {}'.format(ending) +
						' ending in selected dir. Do you want to check for unique names? (If'+
						' multiple tanks are in one folder.)')
				
					if quest == 'yes':
						tanknames = self.get_unique_tank_names(files)
						break
						
				elif nTdtFiles == 0:
					
					messagebox.showinfo('Error..','Tdt files incomplete. Could not find a {} file'.format(ending))
					return
		for tank in tanknames:
		
			tanks[tank] = self.get_tdt_files(files,tankName=tank)
		
		return tanks		


	def get_unique_tank_names(self, files):
		''
		files = [file[:-4] for file in files]
		uniqueTanks = list(set(files))
		
		return uniqueTanks
		
					
	def get_tdt_files(self,files,endings= ['Tbk','tev','Tdx','tsq'], tankName = None):
		''
		if tankName is None:
			return [file for file in files if file.split('.')[-1] in endings]
		else:
				
			tdtFiles =  OrderedDict([(file.split('.')[-1],file) for n,file in enumerate(files) \
					if file.split('.')[-1] in endings and tankName in file])
			return tdtFiles
		
		
	def read_files(self):
		''
		self.tankData = OrderedDict()
		for tankName, fileNames in self.tanks.items():

			data = {}
			
			data['tev'] = self.read_tev(os.path.join(self.dir,fileNames['tev']))
			data['tsq'] = self.read_tsq(os.path.join(self.dir,fileNames['tsq']))
			
				
			self.tankData[tankName] = data	
			print(data)
			
	def read_tev(self,path):
		if os.path.exists(path):
			
			tev = np.memmap(path, mode='r', offset=0, dtype='uint8')
		return tev		

	def read_tsq(self,path):
		''
		if os.path.exists(path):
			
			tsq = np.fromfile(path, dtype=tsq_dtype)
			if tsq[1]['evname'] == chr(EVMARK_STARTBLOCK).encode():
				print(tsq[1]['timestamp'])
				print(time.localtime( tsq[1]['timestamp']))
			else:
				print('no start')
			if tsq[-1]['evname'] == chr(EVMARK_STOPBLOCK).encode():
				print(tsq[-1]['timestamp'])
				print(time.localtime( tsq[-1]['timestamp']))
			else:
				print('No segmet time')
			
			
			            
            
           #  else:
#                 self._seg_t_starts.append(np.nan)
#                 print('segment start time not found')
#             if tsq[-1]['evname'] == chr(EVMARK_STOPBLOCK).encode():
#                 self._seg_t_stops.append(tsq[-1]['timestamp'])
#             else:
#                 self._seg_t_stops.append(np.nan)
#                 print('segment stop time not found')
		return tsq					
		
		
		










