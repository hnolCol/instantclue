"""
	""MULTI - BLOCK ANALYSIS""
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


import pandas as pd
import tkinter as tk 

from sklearn.preprocessing import StandardScaler
from modules.calculations import sgcca 
from modules.dialogs import simple_dialog
from modules.utils import *

import modules.images as img
import webbrowser

from collections import OrderedDict


class sggcaDialog(object):

	def __init__(self, dfClass, sourceTreeView):
	
		
		self.scheme = tk.StringVar(value = 'horst')
		self.dfClass = dfClass
		self.sourceTreeView = sourceTreeView
		self.get_df_ids_and_names() 
		
		if len(self.availableDfs) < 2:
			tk.messagebox.showinfo('Error ..',
				'At least two data frames must be loaded.\nFor single data'+
				' frame supervised analysis you may use PLS-DA or LDA.')
			return
		
		self.blocks = OrderedDict()
		self.blockId = 0
		
		self.build_toplevel() 
		self.build_widgets() 
			
	
	def close(self, event = None):
		'''
		'''			
		self.toplevel.destroy() 	
	
	
	def build_toplevel(self):
		'''
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Sparse Generalized Canonical Correlation Analysis')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.toplevel.bind('<Escape>', self.close)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		 
		self.cont = cont		

	def build_widgets(self):
		'''
		'''
		labelTitle = tk.Label(self.cont, 
					text = 'Sparse Generalized Canonical Correlation Analysis', 
					**titleLabelProperties) 
					
		labelTitle.grid(padx=5,pady=(15,5), columnspan=6 ,sticky=tk.W)	
		
		labelInfo = tk.Label(self.cont, 
					text = 'This method is based on the paper "Variable selection for generalized'+
					' canonical correlation analysis." by  A. Tenenhaus et al. (2014).\n\n'+
					'Please acknowledge their work by a citation.',
					bg = MAC_GREY, justify = tk.LEFT)
		labelInfo.grid(padx=3, sticky=tk.W) 
		labelInfo.bind('<Button-1>', lambda event: webbrowser.open('https://www.ncbi.nlm.nih.gov/pubmed/24550197'))
		make_label_button_like(labelInfo) #stored in utils module
		
		labelInfo2 = tk.Label(self.cont, 
							  text = 'Order of samples must match in blocks!',
							  **titleLabelProperties)
		labelInfo2.grid(row=3,padx=4,sticky=tk.W)
		
		
		_,_,_,self.addImg = img.get_data_upload_and_session_images()
		self.delIcon,_,_ = img.get_delete_cols_images()
		
		addButton = create_button(self.cont, image = self.addImg, command = self.add_block)
		addButton.grid(row=3,padx=4,column=0, sticky=tk.E) 

		self.add_block_headings()
		self.add_block()
		self.add_block()
		
		tk.Label(self.cont,text='Define design and scheme ..',bg=MAC_GREY).grid(row=5,padx=2,pady=2,sticky=tk.W)			
		designButton = ttk.Button(self.cont, text = 'Design', command = self.define_design)
		designButton.grid(row=6,column=0, sticky=tk.W,pady=(5,10))

		schemeCombo = ttk.Combobox(self.cont, textvariable = self.scheme, width = 8,
												values = ['horst','factorial','centroid'])
		schemeCombo.grid(row=6, column = 0,sticky=tk.E,pady=(5,10))
		
		ttk.Separator(self.cont,orient = tk.HORIZONTAL).grid(columnspan=1,sticky=tk.EW,pady=3)
		
		tk.Label(self.cont,text='Tune parameters ..',bg=MAC_GREY).grid(row=8,padx=2,pady=2,sticky=tk.W)
		
		ttk.Button(self.cont, text = '# Comps', command = lambda: tk.messagebox.showinfo('Under construction','This method is currently under construction.')).grid(row=9,column=0,padx=3,pady=1,sticky=tk.W)
		ttk.Button(self.cont, text = 'Sparsity', command = lambda: tk.messagebox.showinfo('Under construction','This method is currently under construction.')).grid(row=9,column=0,padx=3,pady=1,sticky=tk.E)
		
		ttk.Separator(self.cont,orient = tk.HORIZONTAL).grid(row=10,columnspan=1,sticky=tk.EW,pady=3)
				
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close) 
		runButton = ttk.Button(self.cont, text = 'Run', command = self.run) 
		runButton.grid(row = 11, column = 0, sticky=tk.W,pady=5)
		closeButton.grid(row = 11, column = 0, sticky=tk.E,pady=5)
		 

	def add_block_headings(self):
		'''
		'''
		self.blockFrame = tk.Frame(self.cont, bg = MAC_GREY)
		self.blockFrame.grid(sticky=tk.NSEW, columnspan=2) 
		self.blockFrame.grid_columnconfigure(0,weight=1)
		self.cont.grid_columnconfigure(1,weight=1)
		self.cont.grid_rowconfigure(3,weight=1)
		headingFrame = tk.Frame(self.blockFrame, bg = MAC_GREY) 
		headingFrame.grid(sticky=tk.NSEW)
		
		for n,heading in enumerate(['Name','Data Frame','# Comps.',
						'Features\nin columns?','Select\nFeatures', 'Class','Scale Data','l1 constraints']):
		
			lab = tk.Label(headingFrame, text = heading, bg = MAC_GREY)
			lab.grid(row=1,column=n, sticky=tk.EW, pady=3)
			
		ttk.Separator(headingFrame, orient = tk.HORIZONTAL).grid(row=2,sticky=tk.EW,columnspan=8)
		self.adjust_column_width(headingFrame)



	def add_block(self):
		'''
		'''
		blockSetting = dict() 
		
		frame =  tk.Frame(self.blockFrame, bg=MAC_GREY)
		frame.grid(sticky=tk.NSEW)
		
		
		
		ent = ttk.Entry(frame, width = 7) 
		ent.grid(row=1,column=0, sticky=tk.EW, pady=3)
		
		comboDF = ttk.Combobox(frame, width = 8, values = self.availableDfs, state='readonly')
		comboDF.grid(row=1,column=1, sticky=tk.EW, pady=3) 
		comboDF.bind('<<ComboboxSelected>>',lambda event, id = self.blockId: self.change_block_name(event,id=id))

		comboEntry = ttk.Entry(frame, width = 4)
		comboEntry.grid(row=1,column=2, sticky=tk.EW, pady=3)  
		comboEntry.insert(0,'2')
		
		cb = ttk.Checkbutton(frame, text= '')
		cb.state(['!alternate'])
		cb.state(['selected'])
		cb.grid(row=1,column=3, pady=3) 
		
		featuresButton = ttk.Button(frame,
					text = '...',  width = 3,
					command = lambda id=self.blockId: self.define_features(id))
		featuresButton.grid(row=1,column=4, pady=3)		

		classButton = ttk.Button(frame,
								 text = '...', width = 3,
								 command = lambda id=self.blockId: self.define_class(id)) 
								 
		classButton.grid(row=1,column=5, pady=3)
		
		cbScale = ttk.Checkbutton(frame, text = '')
		cbScale.state(['!alternate'])
		cbScale.state(['selected'])	
		cbScale.grid(row=1,column=6, pady=3) 
		
		entryl1Const = ttk.Entry(frame, width = 4)
		entryl1Const.grid(row=1,column=7,pady=3)
		
		
		delButton = create_button(frame,
								image = self.delIcon, 
								command = lambda id = self.blockId: self.remove_block(id=id))
		delButton.grid(row = 1,column = 8, pady = 3, sticky = tk.W)
		
		
		
		
		ttk.Separator(frame, orient = tk.HORIZONTAL).grid(sticky=tk.EW,columnspan=8)								
						
		blockSetting['blockName'] = ent
		blockSetting['dataFrame'] = comboDF
		blockSetting['n_components'] = comboEntry
		blockSetting['featureInCols'] = cb
		blockSetting['scaleData'] = cbScale
		blockSetting['frame'] = frame
		blockSetting['classButton'] = classButton
		blockSetting['featureButton'] = featuresButton
		blockSetting['entryl1Const'] = entryl1Const
	
		
		self.save_block(blockSetting)
				
		if self.blockId <= len(self.availableDfs):
			comboDF.set(self.availableDfs[self.blockId-1])
			self.change_block_name(event=None,blockName=self.availableDfs[self.blockId-1], id = self.blockId-1)	
			
		self.adjust_column_width(frame)
		
		
	def define_class(self,id):
		'''
		'''
		dfString = self.blocks[id]['dataFrame'].get() 
		if dfString not in self.blockIdtoDfId:
			tk.messagebox.showinfo('Error..','Please select a data frame.',parent=self.toplevel)
			return
			
		dfID = self.blockIdtoDfId[dfString]
		
		catColumns = self.dfClass.get_categorical_columns_by_id(dfID)
		dialog = simple_dialog.simpleUserInputDialog(['Class'],
													 [catColumns[0]],
													 catColumns, 
													 'Select Class Columns',
													 'Please define column that holds information about the classes.')
	
		self.blocks[id]['classNames'] = self.dfClass.dfs[dfID][dialog.selectionOutput['Class']].values
		self.blocks[id]['classButton'].configure(text="\u221A")
	
	def define_features(self,id):
	
		dfString = self.blocks[id]['dataFrame'].get() 
		if dfString not in self.blockIdtoDfId:
			tk.messagebox.showinfo('Error..','Please select a data frame.',parent=self.toplevel)
			return
			
		dfID = self.blockIdtoDfId[dfString]
		numColumns = self.dfClass.get_numeric_columns_by_id(dfID)
		
		
		if self.blocks[id]['featureInCols'].instate(['selected']):
		
			pass
			
		else:
			catColumns = self.dfClass.get_categorical_columns_by_id(dfID)
			featureNamesDialog = simple_dialog.simpleUserInputDialog(['Feature Names'],
													 [catColumns[0]],
													 catColumns, 
													 'Select feature names column',
													 'Please select Column that holds feature names (Gene names, Lipid names)')					
		
		
			if len(featureNamesDialog.selectionOutput) == 0:
				return
		
		dialog = simple_dialog.simpleListboxSelection('Select feature columns.',
														  numColumns,
														  'Feature Selection')
		
		if len(dialog.selection) > 0:
			
			featureColumns = dialog.selection 	
			
			blockDF = self.dfClass.dfs[dfID][featureColumns]
			
			if self.blocks[id]['featureInCols'].instate(['selected']):
				
				self.blocks[id]['blockDF'] = blockDF
				self.blocks[id]['featureNames'] = blockDF.columns.values.tolist() 
				
			else:
				
				self.blocks[id]['blockDF'] = blockDF.transpose().values 
				self.blocks[id]['featureNames'] = featureNamesDialog.selectionOutput['Feature Names'] 			
		
		if self.blocks[id]['scaleData'].instate(['selected']):
			
			self.blocks[id]['blockDF'] = StandardScaler().fit_transform(self.blocks[id]['blockDF'])
		
		self.blocks[id]['featureButton'].configure(text="\u221A")

	def run(self):
	
		if hasattr(self,'C') == False:
			tk.messagebox.showinfo('Error ..','Please provide a design.',parent=self.toplevel)
			self.define_design()
			
		
		sgccaSettings, blockNames = self.prepare_data()
		if sgccaSettings is None:
			return
		sgccaSettings['C'] = self.C		
				
		
		SGCCA = sgcca.SGCCA(**sgccaSettings)
		SGCCA.fit()
		
		results = SGCCA.get_result()
		
		n = 0
		dataY = pd.DataFrame()
		
		for id, block in self.blocks.items():
			if n == 0:
				dataY['Classes'] = block['classNames']
				
			data = results['Y'][n]
			blockName = blockNames[n] 
			columns = ['{}_{}'.format(col,blockName) for col in data.columns]
			dat = pd.DataFrame(data.values,columns=columns,index=dataY.index)
			dataY[columns] = dat[columns]
			
			n += 1
			
		id = self.dfClass.get_next_available_id()
		self.dfClass.add_data_frame(dataY,id=id,fileName = 'SGGCA_Y')
		colDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
		self.sourceTreeView.add_new_data_frame(id,'SGGCA_Y',colDataTypeRel)
		
		dataA, dataC = SGCCA.get_non_zero_features(blockNames = blockNames)		
		
		
		mergedDf, corrDfs, corrNames = self.subset_original_data(dataA,sgccaSettings)
		
		
		for corrDf, corrName in zip(corrDfs,corrNames):
			id = self.dfClass.get_next_available_id()
			self.dfClass.add_data_frame(corrDf,id=id,fileName = corrName)
			colDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
			self.sourceTreeView.add_new_data_frame(id,corrName,colDataTypeRel)	
			
					
		id = self.dfClass.get_next_available_id()
		self.dfClass.add_data_frame(mergedDf,id=id,fileName = 'selFeatures')
		colDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
		self.sourceTreeView.add_new_data_frame(id,'selFeatures',colDataTypeRel)				
		
		id = self.dfClass.get_next_available_id()
		self.dfClass.add_data_frame(dataA,id=id,fileName = 'SGGCA_a[OuterWeight]')
		colDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
		self.sourceTreeView.add_new_data_frame(id,'SGGCA_a[OuterWeight]',colDataTypeRel)		
			
		id = self.dfClass.get_next_available_id()
		self.dfClass.add_data_frame(dataC,id=id,fileName = 'SGGCA_corr')
		colDataTypeRel = self.dfClass.get_columns_data_type_relationship_by_id(id)
		self.sourceTreeView.add_new_data_frame(id,'SGGCA_corr',colDataTypeRel)			
		
		
	def subset_original_data(self,dataA,sgccaSettings):
		'''
		'''		
		def find_comps(row):
			return np.where(np.abs(row)>0)[0][0] 
			
		columns = [col for col in dataA.columns if col not in ['Feature','Block']]
		
		dataA['Component'] = dataA[columns].apply(lambda row: find_comps(row), axis=1)
		#dataA['Component'] = dataA['Component'] + 1
		groupedA = dataA.groupby('Block',sort=False)
		
		dfCollection = []
		
		corrCollection = []
		corrBlockNames = []
		
		for id,block in self.blocks.items():
		
			
			dfString = block['dataFrame'].get() 
			dfID = self.blockIdtoDfId[dfString]
			blockName = block['blockName'].get()
			columns = groupedA.get_group(blockName)['Feature'].values.tolist()
			self.dfClass.set_current_data_by_id(dfID)
			df = self.dfClass.get_current_data_by_column_list(columns,ignore_clipping = True)
			
			corrCollection.append(df) 
			corrBlockNames.append(block['blockName'].get())
			
			dfCollection.append(df.transpose())
			
			dfCollection[-1].columns =  ['{}_{}'.format(class_,n) for n,class_ in enumerate(block['classNames'])]
			dfCollection[-1]['Features'] = columns
			dfCollection[-1]['Block'] = [blockName] * len(df.columns)
			dfCollection[-1].index = range(len(dfCollection[-1].index))
			
			
			
		
		corrDfs, corrNames = self.get_correlations_between_blocks(corrCollection, corrBlockNames)
		
		
		mergedDf = pd.concat(dfCollection, ignore_index = True)
		mergedDf['Component'] = dataA['Component']
	
		
		
		return mergedDf, corrDfs, corrNames
	
	def get_correlations_between_blocks(self,corrCollection,blockNames):
		'''
		'''
		dfCollect = pd.DataFrame()
		corrDfs = []
		corrNames = []
		for blockName in blockNames:
			
			idx = blockNames.index(blockName) 
			corrName = 'corr_{}_to_others'.format(blockName)
			corrNames.append(corrName)
			for n,df in enumerate(corrCollection):
				if n != idx:
					
					dfcorr = self.corr2(corrCollection[idx],df)
					if any(col in dfcorr.columns for col in dfCollect.columns):
						columns = ['{}_{}'.format(col,blockNames[n]) for col in dfcorr.columns]
					else:
						columns = dfcorr.columns
					
					dfCollect[columns] = dfcorr
			
			dfCollect['Feature'] = dfCollect.index
			dfCollect.index = range(len(dfCollect.index))
			corrDfs.append(dfCollect)
			dfCollect = pd.DataFrame()
		
		return corrDfs, corrNames
			

			
	def corr2(self,df1, df2):	
		'''
		'''
		n = len(df1)
		v1, v2 = df1.values, df2.values
		sums = np.multiply.outer(v2.sum(0), v1.sum(0))
		stds = np.multiply.outer(v2.std(0), v1.std(0))
		return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
                        df2.columns, df1.columns)			
   		
   	    
    	
    	
    	
    	
		
		
	def prepare_data(self):
		
		sgccaSettings = dict() 
		sgccaSettings['X'] = []
		sgccaSettings['n_components'] = []
		sgccaSettings['featureNames'] = []
		sgccaSettings['scaleData'] = []
		blockNames = []
		l1const = []
		n = 0
		
		for id, blocks in self.blocks.items():
			
			if 'blockDF' not in blocks:
				tk.messagebox.showinfo('Error ..','No Features selected yet.',parent=self.toplevel)
				return None, None
				
			sgccaSettings['X'].append(blocks['blockDF'])
			try:
				sgccaSettings['n_components'].append(int(blocks['n_components'].get()))
			except:
				tk.messagebox.showinfo('Error ..',
								'Could not interpret component input as integer.',
								parent=self.toplevel)
				return None, None
				
			
			sgccaSettings['featureNames'].append(blocks['featureNames'])
			sgccaSettings['scaleData'] = blocks['scaleData'].instate(['selected'])
			
			
			if len(blocks['entryl1Const'].get().split(',')) != 1:
				try:
					l1const.append([float(l1) for l1 in blocks['entryl1Const'].get().split(',')])
				except:
					tk.messagebox.showinfo('Error ..',
											'Could not convert l1 constraint input to float.',
											parent = self.toplevel)
					return	None, None						
			
				if len(l1const[-1]) > sgccaSettings['n_components'][-1]:
					tk.messagebox.showinfo('Error ..',
										   'Number of constraints must match number of components.',
										   parent=self.toplevel)
					return None, None
			else:
				try:
					l1const.append(float(blocks['entryl1Const'].get()))
					if l1const[-1] > 1 or \
					l1const[-1] < 1/np.sqrt(len(sgccaSettings['featureNames'][-1])):
						tk.messagebox.showinfo('Error ..', 
							'l1 must be < 1 and > 1 / # of features',
							parent=self.toplevel)
						return None, None
					
					
				except:
					tk.messagebox.showinfo('Error ..',
											'Could not convert l1 constraint input to float.',
											parent = self.toplevel)
					return None, None
			
			
			if n == 0:
				sgccaSettings['Y'] = blocks['classNames']
				n+=1
			else:
				if np.array_equal(sgccaSettings['Y'],blocks['classNames']) == False:
					tk.messagebox.showinfo('Error ..',
										   'Class labels do not match!', 
										   parent = self.toplevel) 
					return None, None
			blockNames.append(blocks['blockName'].get())
		sgccaSettings['c1'] = np.asarray(l1const)
		print(sgccaSettings['c1'])
			
		return sgccaSettings, blockNames
			
			
			
	def define_design(self):
		
		'''
		'''
		designDialog = defineDesign(self.blocks)
		if hasattr(designDialog, 'C'):
			self.C = designDialog.C
			
			#print(self.C)
			
			

	def remove_block(self,id):
		'''
		Remove a block
		'''
		self.blocks[id]['frame'].destroy() 
		del self.blocks[id]
					
	def save_block(self,blockSettings):
		
		self.blocks[self.blockId] = blockSettings
		self.blockId += 1

	def adjust_column_width(self,frame):
		'''
		'''
		for colNum, width in enumerate([150,150] + [83] * 7):
			if width > 80:
				scale = 1
			else:
				scale = 0
			frame.grid_columnconfigure(colNum, minsize = width, weight=scale)
		

	def get_df_ids_and_names(self):
		'''
		'''
		self.availableDfs = []
		self.blockIdtoDfId = dict()
		for id in self.dfClass.dfs.keys():
			
			fileName = self.dfClass.fileNameByID[id]
			dfString = '{} ID: {}'.format(fileName,id) 
			self.availableDfs.append(dfString)
			self.blockIdtoDfId[dfString] = id
			
	
	def change_block_name(self,event = None, id = None,blockName = None):
		'''
		'''
		
		if id is None:
			return
			
		if blockName is None and event is None:
			return
			
		if event is None:
			name = blockName.split('ID:')[0]
		else:
			name = event.widget.get().split('ID:')[0]
		self.blocks[id]['blockName'].delete(0, tk.END)
		self.blocks[id]['blockName'].insert(0,name)
				
				
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 


class plotHelper(object):
	
	def __init__(self, SGCCA):
	
		self.SGCCA = SGCCA
	
		self.build_toplevel()
		self.build_widgets()
	
		self.toplevel.wait_window()
		
	def close(self):
		
		self.toplevel.destroy()	
	
	def build_toplevel(self):
	
		'''
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('SGCCA Results')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.toplevel.bind('<Escape>', self.close)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		self.cont = cont		
		
	def build_widgets(self):
		'''
		'''
		labelTitle = tk.Label(self.cont, 
					text = 'Visualizing SGCCA Results', 
					**titleLabelProperties) 
					
		labelTitle.grid(padx=5,pady=(15,5), columnspan=6 ,sticky=tk.W)			
		
			
		
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		
		
		


class defineDesign(object):
	
	def __init__(self, blockSettings):
	
		self.blocks = blockSettings
	
		self.build_toplevel()
		self.build_widgets()
	
		
		self.toplevel.wait_window()
		
	def close(self):
		
		self.toplevel.destroy()	
	
	def build_toplevel(self):
	
		'''
		'''
		self.toplevel = tk.Toplevel() 
		self.toplevel.wm_title('Design Creator')
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.toplevel.bind('<Escape>', self.close)
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(expand=True, fill='both')
		cont.grid_rowconfigure(0,weight = 1)
		cont.grid_columnconfigure(1,weight = 1) 
		self.cont = cont		
		
	def build_widgets(self):
		'''
		'''
		labelTitle = tk.Label(self.cont, 
					text = 'Define design for correlation analysis.\n0 - no correlation 1 - strong correlation assumed.', 
					**titleLabelProperties) 
					
		labelTitle.grid(padx=5,pady=(15,5), columnspan=6 ,sticky=tk.W)			
		
		self.build_matrix()
		
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		doneButton = ttk.Button(self.cont, text = 'Done', command = self.apply)
		
		doneButton.grid(row=3, padx=3,pady=2)
		closeButton.grid(column = 1,row=3, padx=3,pady=2)
		
	def build_matrix(self):
	
		frame = tk.Frame(self.cont,bg=MAC_GREY)
		frame.grid(sticky=tk.NSEW, columnspan = 2)
		self.corrEntries = OrderedDict()
		n = 0
		for id, blockSets in self.blocks.items():
			
			labRow = tk.Label(frame,text = blockSets['blockName'].get(),bg=MAC_GREY)
			labRow.grid(row = n+1, column = 0, sticky=tk.EW)
			
			labCol = tk.Label(frame, text = blockSets['blockName'].get(),bg=MAC_GREY)
			labCol.grid(row = 0, column = n+1, sticky=tk.EW)
			self.corrEntries[n] = []
			for p in range(len(self.blocks)-(n+1)):
			
				ent = ttk.Entry(frame,width=4, justify = 'center')
				ent.grid(row=n+1, column = (p+1) + (n + 1), sticky=tk.EW)
				ent.insert(0,0.1)
				self.corrEntries[n].append(ent)
			
			labInBlock = tk.Label(frame, text = '0',bg=MAC_GREY)
			labInBlock.grid(row = n+1, column = n+1, sticky=tk.EW)
			
			frame.grid_rowconfigure(n+1,weight=1,minsize=40)
			frame.grid_columnconfigure(n+1,weight=1,minsize=40)
			n += 1
		
		#for m in range(n):
		#	ttk.Separator(frame,orient = tk.HORIZONTAL).grid(row=n,columnspan=n+1,sticky=tk.EW)
		#	ttk.Separator(frame).grid(column=n, rowspan=n+1,sticky=tk.NS)
			
	def apply(self):
	
		n_blocks = len(self.blocks)
		C = np.zeros((n_blocks,n_blocks))
		
		for n_row,values in self.corrEntries.items():
			try:
				vals = np.asarray([float(x.get()) for x in values])
			except:
				tk.messagebox.showinfo('Error ..',
					'Input could not be converted to float.',
					parent=self.toplevel)
				return
				
			C[n_row,(n_row+1):n_blocks] = vals
			C[(n_row+1):n_blocks,n_row] = vals
		
		self.C = C
		self.close()
		
					
		
	













