

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
tbk_field_types = [
    ('StoreName', 'S4'),
    ('HeadName', 'S16'),
    ('Enabled', 'bool'),
    ('CircType', 'int'),
    ('NumChan', 'int'),
    ('StrobeMode', 'int'),
    ('TankEvType', 'int32'),
    ('NumPoints', 'int'),
    ('DataFormat', 'int'),
    ('SampleFreq', 'float64'),
]
data_formats = {
    0: 'float32',
    1: 'int32',
    2: 'int16',
    3: 'int8',
    4: 'float64',
}
_signal_channel_dtype = [
    ('name', 'U64'),
    ('id', 'int64'),
    ('sampling_rate', 'float64'),
    ('dtype', 'U16'),
    ('units', 'U64'),
    ('gain', 'float64'),
    ('offset', 'float64'),
    ('group_id', 'int64'),
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
import re
from collections import OrderedDict

class TDTTankToPandas(object):
	
	
	def __init__(self,dir):
		''
		self.signal_data_buff = {}
		self.samp_per_chunk = {}
		self.signal_channels = []
		self.sig_sample_per_chunk = {}
		self.set_t_starts = []
		self.set_t_stop = []
		self.times = {}
		self.signal_index = {}
		self.sig_length = {}
		self.sig_dtype_by_group = {}
		
		self.header = {}
		self.dir = dir
		
		self.check_dir(dir)
		
		if hasattr(self,'tanks'):
			
			return self.read_files()
		
		
	def check_dir(self,dir):
		''
		tanknames = [os.path.basename(self.dir)]
		
		files = os.listdir(self.dir)
		files = self.get_tdt_files(files)
		
		self.tanks = self.check_files(files,tanknames = tanknames)
		
		#if self.tanks is None:
		#	return

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
			self.signal_index[tankName] = {}
			
			data['tev'] = self.read_tev(os.path.join(self.dir,fileNames['tev']))
			data['tsq'] = self.read_tsq(os.path.join(self.dir,fileNames['tsq']), tankName)
			data['tbk'] = self.read_tbk(os.path.join(self.dir,fileNames['Tbk']))
			data['keep'] = data['tbk']['TankEvType'] == EVTYPE_STREAM
			
			
			self.tankData[tankName] = data
		
		for tankName, fileNames in self.tanks.items():
		
			self.extract_data(self.tankData[tankName], tankName)	
		#for tankName , fileNames in self.tanks.items():
		#	rawSignal, columnNames = self.get_raw_signal('all',tankName)
		#return rawSignal, columnNames
			
	def read_tev(self,path):
		if os.path.exists(path):
			
			tev = np.memmap(path, mode='r', offset=0, dtype='uint8')
			#print('TEV SHAPE',tev.shape)
			
		return tev		

	def read_tsq(self,path, tankName):
		''
		if os.path.exists(path):
			
			tsq = np.fromfile(path, dtype=tsq_dtype)
			if tsq[1]['evname'] == chr(EVMARK_STARTBLOCK).encode():
			#	print(tsq[1]['timestamp'])
			#	print(time.localtime( tsq[1]['timestamp']))
				self.set_t_starts.append(tsq[1]['timestamp'])
			else:
				print('no start')
			if tsq[-1]['evname'] == chr(EVMARK_STOPBLOCK).encode():
			#	print(tsq[-1]['timestamp'])
			#	print(time.localtime( tsq[-1]['timestamp']))
				self.set_t_stop.append(tsq[-1]['timestamp'])
			else:
				print('No segmet time')

			#print(tsq[-1]['timestamp']-tsq[1]['timestamp'])
			self.times[tankName] = tsq[-1]['timestamp']-tsq[1]['timestamp']
			 
		return tsq					
		
	def read_tbk(self,fileName):
		'''
		'''
		
		with open(fileName, mode='rb') as f:
			txt_header = f.read() 
		
		props = []
		for chan_grp_header in txt_header.split(b'[STOREHDRITEM]'):
			if chan_grp_header.startswith(b'[USERNOTEDELIMITER]'):
				break
			i = OrderedDict()
			pattern = b'NAME=(\S+);TYPE=(\S+);VALUE=(\S+);'
			r = re.findall(pattern, chan_grp_header)
			for name, _type, value in r:
				i[name.decode('ascii')] = value
			props.append(i)
		info_channel_groups = np.zeros(len(props), dtype=tbk_field_types)
		for i, info in enumerate(props):
			for k, dt in tbk_field_types:
				v = np.dtype(dt).type(info[k])
				info_channel_groups[i][k] = v
		return info_channel_groups

	
	def extract_data(self,data, tankName = None):
		''
		
		self.signal_data_buff[tankName] = {}
		self.sig_length[tankName] = {}
		signal_channels = []
		for group_id, info in enumerate(data['tbk'][data['keep']]):
			self.sig_sample_per_chunk[group_id] = info['NumPoints']
			for c in range(info['NumChan']):
				chan_index = len(signal_channels)
				chan_id = c + 1
				chan_name = '{} {}'.format(info['StoreName'], c + 1)
				smapling_rate = None
				dtype = None
				#for id, tankData in self.tankData.items():
				tsq = data['tsq']

				mask = (tsq['evtype'] == EVTYPE_STREAM) & \
                           (tsq['evname'] == info['StoreName']) & \
                           (tsq['channel'] == chan_id)
				data_index = tsq[mask].copy()
				self.signal_index[tankName][chan_index] = data_index
				dtype = data_formats[data_index['dataformat'][0]]
				t_start = data_index['timestamp'][0]
				sampling_rate = float(data_index['frequency'][0])
				size = info['NumPoints'] * data_index.size
				self.sig_length[tankName][group_id] = size
				self.sig_sample_per_chunk[tankName] = info['NumPoints']
				self.sig_dtype_by_group[group_id] = np.dtype(dtype)
				self.signal_data_buff[tankName][chan_index] =  data['tev']
				signal_channels.append((chan_name,chan_id, sampling_rate, dtype,'V',1,0, group_id))	
			
		self.signal_channels = np.array(signal_channels,dtype=_signal_channel_dtype)
        
	def get_tank_names(self):
		'''
		'''
		return list(self.tanks.keys())				
			
	def get_raw_signal(self,channels = 'all', tankName = None):
		'''
		'''
		if tankName is None:
			tankName = list(self.tanks.keys())[0]
		
		channelNames = []
		for tankName, fileNames in self.tanks.items():
			channels = [0,1,2]
			data = self.tankData[tankName]
			startIdx = 0
			group_id = self.signal_channels[channels[0]]['group_id']
			
			stopIdx = self.sig_length[tankName][group_id]
			rawSignal = np.zeros((stopIdx-startIdx,len(channels)))
			sample_per_chunk = self.sig_sample_per_chunk[group_id]

			b0 = startIdx // sample_per_chunk
			b1 = int(np.ceil(stopIdx / sample_per_chunk))
			
			dt = self.sig_dtype_by_group[group_id]
			chunk_nb_bytes = sample_per_chunk * dt.itemsize
			for c, channelIdx in enumerate(channels):
				data_index = self.signal_index[tankName][channelIdx]
				data_buf = self.signal_data_buff[tankName][channelIdx]
				ind  = 0
				channelNames.append(self.signal_channels[channels[c]]['name'])
				for bl in range(b0,b1):
					ind0 = data_index[bl]['offset']
					ind1 = ind0 + chunk_nb_bytes
					data = data_buf[ind0:ind1].view(dt)
					
					if bl == b1-1:
						border = data.size - (stopIdx  % sample_per_chunk)
						data = data[:-border]
					if bl == b0:
						border = startIdx % sample_per_chunk
						data = data[border:]
					rawSignal[ind:data.size + ind,c] = data
					ind += data.size
			timeColumn = self.times[tankName]
			return rawSignal, channelNames, timeColumn, tankName
				






