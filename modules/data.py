"""
	""DATA MANAGEMENT IN INSTANT CLUE""
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

"""
Principles
=============
Data frames are stored by an internal id in a dict. 
Columns added are automatically tested for duplicates.
Simple calculations on data are performed in this class as well.

Live Filtering 

Maskig a certain part of the data frame using the live filter 
activity is performed like:
 - store boolean array in dict that has the same id as the data frame
 - data for activities are received by get_current_data, get_current_data_by_id,
   or get_current_data_by_columnList. If the id is present in the masking
   dict, data are returned in a trunkated way. 
   Note that this can always be de-activated by "ignoreMasking = True".
   The liver filter is a dialog window. If closed, all masking is lost.
"""

import numpy as np
from scipy.signal import lfilter
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import scale
from itertools import compress, chain
import pandas as pd


from collections import OrderedDict
from modules.dialogs.categorical_filter import categoricalFilter
from modules.utils import *
import time

def z_score(x):
	mean = x.mean()
	std = x.std() 
	vector = (x-mean)/std
	return vector
   
       
calculations = {'log2':np.log2,
					'-log2':lambda x: np.log2(x)*(-1),
					'log10':np.log10,
					'-log10':lambda x: np.log10(x)*(-1),
					'ln':lambda x: np.log(x),
					'Z-Score':lambda x: z_score(x),
					}	
	


class DataCollection(object):
	'''
	'''

	def __init__(self, workflow = None):
	
		self.currentDataFile = None
		self.workflow = workflow
		self.df = pd.DataFrame()
		self.df_columns = []
		self.dataFrameId = 0
		self.dfs = OrderedDict()
		self.dfsDataTypesAndColumnNames = OrderedDict() 
		self.fileNameByID = OrderedDict() 
		self.rememberSorting = dict()
		self.replaceObjectNan = '-'
		self.clippings = dict()
		self.droppedRows = dict() 
	
	
	def add_data_frame(self,dataFrame, id = None, fileName = '', 
							sourceBranch = None, addInfo = {}):
		'''
		Adds new dataFrame to Dict.
		'''
		if id is None:
			id = self.get_next_available_id()
				
		self.extract_data_type_of_columns(dataFrame,id)
		self.dfs[id] = dataFrame
		self.rename_data_frame(id,fileName)
		rows,columns = self.dfs[id].shape
		branchInfo = OrderedDict([('Name: ',fileName),
								  ('ID: ',id),
								  ('Columns: ',str(columns)),
								  ('Rows: ',str(rows))])
		
		self.workflow.add_branch(id, fileName, branchInfo,sourceBranch,addInfo)
	
	
	def add_count_through_column(self, columnName = None):
		'''
		Simply adds a column that enumerates over the data.
		'''
		if columnName is None:
			columnName = 'CountThrough'
		nRow = self.get_row_number() 
		countThrough = np.arange(0,nRow)
		columnName = self.evaluate_column_name(columnName=columnName)
		
		self.df[columnName] = countThrough
		self.df[columnName] = self.df[columnName].astype('int64')
		self.update_columns_of_current_data()
		return columnName

	
	def add_column_to_current_data(self,columnName,columnData,evaluateName = True):
		'''
		Adds a new column to the current data
		'''
		if evaluateName:
			columnName = self.evaluate_column_name(columnName)
		self.df.loc[:,columnName] = columnData
		self.update_columns_of_current_data() 
	
		
	def calculate_kernel_density_estimate(self,numericColumns):
		'''
		'''
		newColumnName = 'kde_{}'.format(numericColumns)
		data = self.df[numericColumns].dropna(subset=numericColumns)
		indexSubset = data.index
		nRows = len(indexSubset)
		nColumns = len(numericColumns) 
		bandwidth = nRows**(-1/(nColumns+4)) 
		
		kde = KernelDensity(bandwidth=bandwidth,
                        kernel='gaussian', algorithm='ball_tree')
		
		kde.fit(data) 
		kde_exp = pd.DataFrame(np.exp(kde.score_samples(data)),columns = [newColumnName], index=indexSubset)
		self.df = self.df.join(kde_exp)
		self.update_columns_of_current_data() 
		self.save_current_data()
		return newColumnName
	
         
         
		
	def calculate_rolling_metric(self,numericColumns,windowSize,metric,quantile = 0.5):
		'''
		Calculates rolling windows and metrices (like mean, median etc). 
		Can be used for smoothing
		'''
		newColumnNames = ['[{}_w{}]_{}'.format(metric,windowSize,columnName) for columnName in numericColumns] 
		rollingWindow = self.df[numericColumns].rolling(window=windowSize)
		
		if metric == 'mean':
			self.df[newColumnNames] = rollingWindow.mean() 
		elif metric == 'median':
			self.df[newColumnNames] = rollingWindow.median()
		elif metric == 'sum':
			self.df[newColumnNames] = rollingWindow.sum() 
		elif metric == 'max':
			self.df[newColumnNames] = rollingWindow.max()
		elif metric == 'min':
			self.df[newColumnNames] = rollingWindow.min()
		elif metric == 'std':
			self.df[newColumnNames] = rollingWindow.std()
		elif metric == 'quantile':
			self.df[newColumnNames] = rollingWindow.quantile(quantile=quantile)
		
		self.update_columns_of_current_data()
		
		return newColumnNames
		

	def calculate_row_wise_metric(self,metric,numericColumns,promptN):
		'''
		'''

		if metric == 'Mean & Stdev [row]':
			newColumnName = ['{}_{}'.format(metric,get_elements_from_list_as_string(numericColumns))\
			 for metric in ['Mean','Stdev']]
		elif metric == 'Mean & Sem [row]':
			newColumnName = ['{}_{}'.format(metric,get_elements_from_list_as_string(numericColumns))\
			 for metric in ['Mean','Sem']]
		elif metric == 'Square root [row]':
			newColumnName = ['{}_{}'.format(metric.replace(' [row]','')	,columnName) for columnName in numericColumns]
		elif metric in ['N ^ x [row]','x ^ N [row]','x * N [row]']:
			newColumnName = ['{}({})_{}'.format(metric.replace(' [row]','')	,promptN,columnName) for columnName in numericColumns]
		else:
			newColumnName = '{}_{}'.format(metric.replace(' [row]','')	,get_elements_from_list_as_string(numericColumns))
		if metric == 'Mean [row]':
			self.df[newColumnName] = self.df[numericColumns].mean(axis=1)
		elif metric == 'Square root [row]':
			self.df[newColumnName] = self.df[numericColumns].apply(np.sqrt,axis=1) 
		elif metric == 'Stdev [row]':
			self.df[newColumnName] = self.df[numericColumns].std(axis=1)
		elif metric == 'Sem [row]':
			self.df[newColumnName] = self.df[numericColumns].sem(axis=1)
		elif metric == 'Median [row]':
			self.df[newColumnName] = self.df[numericColumns].median(axis=1)
		elif metric == 'x * N [row]':
			self.df[newColumnName] = self.df[numericColumns] * promptN
		elif metric == 'N ^ x [row]':
			self.df[newColumnName] = self.df[numericColumns].apply(lambda row, pow=promptN: np.power(pow,row),axis=1)
		elif metric == 'x ^ N [row]':
			self.df[newColumnName] = self.df[numericColumns]** promptN# .apply(lambda row, pow=promptN: np.power(row,pow), axis=1)
		elif metric == 'Mean & Stdev [row]':
			self.df[newColumnName[0]] = self.df[numericColumns].mean(axis=1)
			self.df[newColumnName[1]] = self.df[numericColumns].std(axis=1)
		elif metric == 'Mean & Sem [row]':
			self.df[newColumnName[0]] = self.df[numericColumns].mean(axis=1)
			self.df[newColumnName[1]] = self.df[numericColumns].sem(axis=1)
		
		if isinstance(newColumnName,str):
			newColumnName = [newColumnName]
			
		self.update_columns_of_current_data()

		return newColumnName		 
		
	

	def change_data_type_in_current_data(self,columnList, newDataType):
		'''
		Changes the DataType of a List of column Names
		'''
		if isinstance(newDataType, list) and len(newDataType) == len(columnList):
			newDataType = dict([(k,v) for k,v in zip(columnList,newDataType)])
		try:		
			self.df[columnList] = self.df[columnList].astype(newDataType)
		except ValueError:
			return 'ValueError'
			
		if newDataType in ['object','str']:
			self.df[columnList].fillna(self.replaceObjectNan,inplace=True)
			
		self.update_columns_of_current_data()
		return 'worked'
	
		
	def combine_columns_by_label(self,columnLabelList, sep='_'):
		'''
		'''
		combinedRowEntries = []
		
		for columnName in columnLabelList[1:]:
			combinedRowEntries.append(self.df[columnName].astype(str).values.tolist())
			
		columnName = 'Comb.: '+str(columnLabelList)[1:-1]
		columnNameEval = self.evaluate_column_name(columnName)
		self.df.loc[:,columnNameEval] = self.df[columnLabelList[0]].astype(str).str.cat(combinedRowEntries,sep=sep)
		
		self.update_columns_of_current_data()
		return columnNameEval

		
	def create_clipping(self, dataID, rowIdxBool):
		'''
		data ID - ID that was given data frame when added to this class
		rowIdx - rows to be temporariley kept.
		'''
 	
		self.clippings[dataID] = rowIdxBool	


	def create_categorical_modules(self, categoricalColumn, aggregate = 'median', 
									sepString =';', id = None, progressBar = None):
		'''
		Input
		Parameter 
		=============
		- categoricalColumn
		- aggregate - method to aggregate 
		- sepString - split string to find unique categories 
		- progressBar - progressBar object to put information out to the user.
		
		Output 
		=============
		'''
		if id is not None:
			self.set_current_data_by_id(id)
		
		if aggregate not in ['mean','median','sum']:
			return None, None
		
		if isinstance(categoricalColumn,str):	
			
			categoricalColumn = [categoricalColumn]
			
		if progressBar is None:
			progressBar.update_progressbar_and_label(0,'Find unique categories ..')
						
		splitData = self.df[categoricalColumn[0]].astype('str').str.split(sepString).values
		flatSplitData = list(set(chain.from_iterable(splitData)))
				
				
		numericColumns = self.dfsDataTypesAndColumnNames[self.currentDataFile]['float64']
		df = pd.DataFrame() 
		collData = []
		
		if progressBar is not None:
			nUniqueCats = len(flatSplitData)
			progressBar.update_progressbar_and_label(0,'Found {} categories ..'.format(nUniqueCats))
		
		for n,category in enumerate(flatSplitData):
			
			regExp = categoricalFilter.build_regex('',
												[category],
												splitString = sepString)
			
			boolIndicator = self.df[categoricalColumn[0]].str.contains(regExp)
			dfSubset = self.df.loc[boolIndicator,:]
			
			if aggregate == 'median':
				aggData = dfSubset[numericColumns].median(axis=0)
			elif aggregate == 'mean':
				aggData = dfSubset[numericColumns].mean(axis=0)
			elif aggregate == 'sum':
				aggData = dfSubset[numericColumns].sum(axis=0)

			
			if progressBar is not None and n % 30 == 0:
				progressBar.update_progressbar_and_label(n/nUniqueCats * 100,
											'Found {} categories .. {}/{}'.format(nUniqueCats,n,nUniqueCats))
			
			collData.append(pd.Series(np.append(aggData,np.array(category))))
				
		df = df.append(collData,ignore_index=True)
		df.columns = numericColumns + categoricalColumn
		df[numericColumns] = df[numericColumns].astype(np.float64)
		if progressBar is not None:
				progressBar.update_progressbar_and_label(100,
											'Done .. ')
				progressBar.close()
		id = self.get_next_available_id()
		fileName = 'Cat. Modules - {}'.format(categoricalColumn[0])
		self.add_data_frame(df,id,fileName) 
		
		return id, fileName			
		

	def reset_clipping(self,dataID):
		'''
		Removes all clippings made
		'''
		if dataID in self.clippings:
			del self.clippings[dataID]
		
	def count_valid_values(self, numericColumnList):
		'''
		Input
		Parameter 
		===========
		numericColumnList - list of numeric columns. 
		
		Output 
		===========
		Evaluated Column name
		'''
		columnName = get_elements_from_list_as_string(numericColumnList, addString =  'valid_values: ')
		columnNameEval = self.evaluate_column_name(columnName)
		validValues = self.df[numericColumnList].count(axis='columns')
		self.df[columnNameEval] = validValues
		self.update_columns_of_current_data()
		return columnNameEval
		
	def delete_rows_by_index(self, index):
		'''
		Delete rows by index
		'''
		self.df.drop(index, inplace=True)
		self.save_current_data()
			
		
	def delete_column_by_index(self, index):
		'''
		Deletes columns of current df by index
		'''
		colname = self.df_columns[index]
		self.df.drop([colname], axis=1, inplace=True) 
		self.update_columns_of_current_data()
		return
		
		
	def delete_columns_by_label_list(self,columnLabelList):
		'''
		Delete columns by Column Name List. 
		'''
		self.df.drop(columnLabelList,axis=1,inplace=True)
		self.update_columns_of_current_data()
				
	def delete_data_file_by_id(self,id):
		'''
		Deletes DataFile by id
		'''	
		if id not in self.dfs:
			return
		
		del self.dfs[id]
		del self.fileNameByID[id]
		del self.dfsDataTypesAndColumnNames[id]
		
		if id == self.currentDataFile:
			self.df = pd.DataFrame()
			self.df_columns = []

	def divide_columns_by_value(self,columnAndValues,baseString):
		'''
		columnAndValues - dict - keys = columns, values = value
		'''
		newColumnNames = []
		for column, correctionValue in columnAndValues.items():
			name = '{} {}'.format(baseString,column)
			newColumnName = self.evaluate_column_name(name)
			self.df[name] = self.df[column] / correctionValue
			newColumnNames.append(newColumnName) 
		self.update_columns_of_current_data()
		return newColumnNames
	
	def substract_columns_by_value(self,columnAndValues,baseString):
		'''
		columnAndValues - dict - keys = columns, values = value
		'''
		newColumnNames = []
		for column, correctionValue in columnAndValues.items():
			name = '{} {}'.format(baseString,column)
			newColumnName = self.evaluate_column_name(name)
			self.df[name] = self.df[column] - correctionValue
			newColumnNames.append(newColumnName) 
		self.update_columns_of_current_data()
		return newColumnNames
				
	def drop_rows_with_nan(self,columnLabelList,how,thresh=None):
		'''
		Drops rows with NaN
		'''
		if isinstance(columnLabelList,list):
			pass
		elif isinstance(columnLabelList,str):
			columnLabelList = [columnLabelList]
		#import time 
		#t1 = time.time() 
		if len(columnLabelList) == 1:
			bool = np.invert(np.isnan(self.df[columnLabelList].values)) 
			dfnoNaN = self.df[bool.tolist()]
		
		else:
		
			dfnoNaN = self.df.dropna(how = how,subset=columnLabelList, thresh=thresh)
		
		if len(self.df.index) > 10000:
		
			idxNaN_txt = '\nDf to big to save. Cannot be undone.\n'
		
		else:
			boolIn = np.isin(self.df.index,dfnoNaN.index)
			idxNaN = self.df.index[boolIn == False]
			idxNaN_txt = np.sum(boolIn == False)
			self.save_dropped_rows(self.currentDataFile,idxNaN)
		
		self.df = self.dfs[id] = dfnoNaN
		self.update_columns_of_current_data()
		#print(time.time()-t1)
		
		return idxNaN_txt
				
			
	def save_dropped_rows(self,id=None,rowIdx=None,reverse=False):
		'''
		Saves rows that were removed from the data set.
		'''
		if id != self.currentDataFile:
			self.set_current_data_by_id(id)
		
		if reverse and id in self.droppedRows:
			
			dfToAdd = self.droppedRows[id][-1]
			del self.droppedRows[id][-1]
			if len(self.droppedRows[id]) == 0:
				del self.droppedRows[id]
				
			self.df = self.df.append(dfToAdd)
			self.dfs[id] = self.df			
			
			
		else:
		
			if id not in self.droppedRows:
		
				self.droppedRows[id] = [self.df.loc[rowIdx,:]]
			else:
				self.droppedRows[id].append(self.df.loc[rowIdx,:])
		
		
		
					
		
	def duplicate_columns(self,columnLabelList):
		'''
		Duplicates a list of columns and inserts the column at the position + 1
		of the original column. 
		'''
		columnLabelListDuplicate = ['Dupl_'+col for col in columnLabelList]
		columnIndexRaw = [self.df_columns.index(col)  for col in self.df_columns if col in columnLabelList]
		
		for i,columnIndex in enumerate(columnIndexRaw):
			columnIndex = columnIndex + 1 + i
			columnColumn = columnLabelListDuplicate[i]
			columnLabel = columnLabelList[i]
			newColumnData = self.df[columnLabel]
			self.insert_column_at_index(columnIndex, columnColumn, newColumnData)
			
		self.update_columns_of_current_data()
		return columnLabelListDuplicate	
	
	def substract_columns_by_column(self,columnLabelList, divColumn):
		'''
		
		'''
		columnNames = []
		if isinstance(divColumn,str):
		
				for column in columnLabelList:
					
					if divColumn in self.df.columns:
						columnName = self.evaluate_column_name('{}-{}'.format(column,divColumn))
						self.df[columnName] = self.df[column] - self.df[divColumn]
						columnNames.append(columnName)
						
		elif isinstance(divColumn,list):
			
				if len(divColumn) != len(columnLabelList):
					return
				
				for column, divColumn in zip(columnLabelList,divColumn):
					columnName = self.evaluate_column_name('{}-{}'.format(column,divColumn))
					self.df[columnName] = self.df[column] - self.df[divColumn]
					columnNames.append(columnName)
				
		self.update_columns_of_current_data()
		return columnNames

	def divide_columns_by_column(self,columnLabelList, divColumn):
		'''
		
		'''
		
		columnNames = []
		if isinstance(divColumn,str):
		
				for column in columnLabelList:
					
					if divColumn in self.df.columns:
						columnName = self.evaluate_column_name('{}/{}'.format(column,divColumn))
						self.df[columnName] = self.df[column] / self.df[divColumn]
						columnNames.append(columnName)
						
		elif isinstance(divColumn,list):
			
				if len(divColumn) != len(columnLabelList):
					return
				
				for column, divColumn in zip(columnLabelList,divColumn):
					#print(column,divColumn)
					columnName = self.evaluate_column_name('{}/{}'.format(column,divColumn))
					self.df[columnName] = self.df[column] / self.df[divColumn]
					columnNames.append(columnName)
				
		self.update_columns_of_current_data()
		return columnNames
			
			
		
	def evaluate_columnNames_of_df(self, df, useExact = True):
		'''
		Checks each column name individually to avoid same naming and overriding.
		'''
		columns = df.columns.values.tolist() 
		evalColumns = [self.evaluate_column_name(column,useExact=useExact) for column in columns]
		df.columns = evalColumns
		return df
		
			
	def evaluate_column_name(self,columnName,columnList = None, useExact = False, maxLength = 80):
		'''
		Check if the column name already exists and how often. Adds a suffix.
		'''
		if columnList is None:
			columnList = self.df_columns 
		
		if len(columnName) > maxLength-10:
			columnName = columnName[:maxLength-30]+'___'+columnName[-30:]
			
		if useExact:
			columnNameExists = [col for col in columnList if columnName == col]
		else:
			columnNameExists = [col for col in columnList if columnName in col]
		
		numberColumnNameExists = len(columnNameExists)
		if numberColumnNameExists > 0:
			newColumnName = columnName+'_'+str(numberColumnNameExists)
		else:
			newColumnName = columnName		
		
		return newColumnName
	
	def extract_data_type_of_columns(self,dataFrame,id):
		'''
		Saves the columns name per data type. In InstantClue there is no difference between
		objects and others non float, int, bool like columns.
		'''
		dataTypeColumnRelationship = dict() 
		for dataType in ['float64','int64','object','bool']:
			try:
				if dataType != 'object':
					dfWithSpecificDataType = dataFrame.select_dtypes(include=[dataType])
				else:
					dfWithSpecificDataType = dataFrame.select_dtypes(exclude=['float64','int64','bool'])
			except ValueError:
				dfWithSpecificDataType = pd.DataFrame() 		
			columnHeaders = dfWithSpecificDataType.columns.values.tolist()
			dataTypeColumnRelationship[dataType] = columnHeaders
		self.dfsDataTypesAndColumnNames[id] = dataTypeColumnRelationship
		
	
		
	def export_current_data(self,format='txt',path = ''):
		'''
		
		'''
		if format == 'txt':
			self.df.to_csv(path, index=None, na_rep ='NaN', sep='\t')
		elif format == 'Excel':
			self.df.to_excel(path, index=None, sheet_name = sheet_name, na_rep = 'NaN')
		

	def export_data_by_id(self, id, format):
		'''
		Set current data and then export the current data.
		'''
		
		self.set_current_data_by_id(id=id)
		self.export_current_data(format=format) 		

	def factorize_column(self,columnLabelList):
		'''
		Factorizes categories in columns. 
		'''
		if isinstance(columnLabelList,list) == False:
			columnLabelList = [columnLabelList]
		dfLabels = pd.DataFrame(columns = ['Category','Factors'])
		newColumnNames = [self.evaluate_column_name('{}_fact'.format(col), useExact = True) for col in columnLabelList]
		for n,column in enumerate(columnLabelList):
			
			y,yLabels = pd.factorize(self.df[column].values)
			df = pd.DataFrame() 
			df['Category_{}'.format(column)] = yLabels
			df['Factors_{}'.format(column)] = list(range(yLabels.size))
			if n == 0:
				dfLabels = df
			else:
				dfLabels = pd.concat([dfLabels,df], axis = 1, ignore_index = True)
			
			self.df[newColumnNames[n]] = y
		dfColumns = [('Category_{}'.format(column),'Factors_{}'.format(column)) for column in columnLabelList]
		dfLabels.columns = list(sum(dfColumns, ()))
		return newColumnNames, dfLabels
		
		
	def fill_na_in_columnList_by_rowMean(self,columnLabelList):
		'''
		'''
		self.df[columnLabelList] = \
		self.df[columnLabelList].apply(lambda row: row.fillna(row.mean()), axis=1)
	
	def fill_na_in_columnList(self,columnLabelList,naFill = None):
		'''
		Replaces nan in certain columns by value
		'''
		if naFill is None:
			naFill = self.replaceObjectNan
		self.df[columnLabelList] = self.df[columnLabelList].fillna(naFill)
	
	
	def fill_na_with_data_from_gauss_dist(self,columnLabelList,downshift,width):
		'''
		Replaces nans with random samples from standard distribution. 
		'''
		means = self.df[columnLabelList].mean()
		stdevs =  self.df[columnLabelList].std()
		for n,numericColumn in enumerate(columnLabelList):
			data = self.df[numericColumn].values
			mu, sigma = means[n], stdevs[n]
			newMu = mu - sigma * downshift
			newSigma = sigma * width
			mask = np.isnan(data)
			data[mask] = np.random.normal(newMu, newSigma, size=mask.sum())
			self.df[numericColumn] = data
		
	def fit_transform(self,obj,columns,namePrefix = 'Scaled'):
		'''
		Fit and transform data using an object from the scikit library.
		'''
		newColumnNames = [self.evaluate_column_name('{}{}'.format(namePrefix,column), useExact = True)
						 	for column in columns]
		
		df, idx = self.row_scaling(obj,columns)
		df_ = pd.DataFrame(df,index = idx, columns = newColumnNames)
		newColumnNames = self.join_df_to_currently_selected_df(df_, exportColumns = True)
		return newColumnNames
		#return
		collectDF = pd.DataFrame()
		for n,column in enumerate(columns):
			X = self.df[column].dropna()
			normX = getattr(obj,'fit_transform')(X.values.reshape(-1, 1))
			df = pd.DataFrame(normX,index = X.index, columns = [newColumnNames[n]])
			
			if len(collectDF.index) == 0:
				collectDF = df
			else:
				collectDF = pd.concat([collectDF,df],axis=1)
		newColumnNames = self.join_df_to_currently_selected_df(collectDF, exportColumns = True)
		return newColumnNames
		
		#print(collectDF)
			
	def row_scaling(self,obj,columnNames):
		"""
		"""
		data = self.df[columnNames].dropna()
		X = np.transpose(data.values)
		normX = getattr(obj,'fit_transform')(X)
		return np.transpose(normX), data.index
		
		
	
	
	def get_categorical_columns_by_id(self,id = None):
		'''
		Returns columns names that are objects
		'''
		if id is None:
			return []
			
		catColumns = self.dfsDataTypesAndColumnNames[id]['object'] 
		
		return catColumns		     	
	
	def get_numeric_columns_by_id(self,id):
		'''
		Returns columns names that are float and integers
		'''
		numColumns = self.dfsDataTypesAndColumnNames[id]['float64'] + \
		self.dfsDataTypesAndColumnNames[id]['int64']
		
		return numColumns			
	def get_numeric_columns(self):
		'''
		Returns columns names that are float and integers
		'''
		return self.get_numeric_columns_by_id(self.currentDataFile) 
		
	def get_data_as_list_of_tuples(self, columns, data = None):
		'''
		Returns data as list of tuples. Can be used for Lasso contains events.
		'''
		if len(columns) < 2:
			return
		if data is None:
			data = self.df
		tuples = list(zip(data[columns[0]], data[columns[1]]))
		return tuples
	
						
	def get_columns_data_type_relationship(self):
		'''
		Returns columns datatypes relationship
		'''
		
		return self.dfsDataTypesAndColumnNames[self.currentDataFile]
		
	def get_columns_data_type_relationship_by_id(self, id):
		'''
		Returns columns datatypes relationship by ID
		'''
		return self.dfsDataTypesAndColumnNames[id]	
	
	
	def get_data_types_for_list_of_columns(self,columnNameList):
		'''
		'''
		dataTypeList = [self.df[column].dtype for column in columnNameList]
		return dataTypeList
			
	def get_complete_data_collection(self):
		'''
		Returns an orderedDictionary with all added data.
		'''
		
		self.save_current_data()
		return self.dfs 
	
	def get_file_names(self):
		'''
		Returns the available file names
		'''
		return list(self.fileNameByID.values())
		
	def get_number_of_columns_in_current_data(self):
		'''
		Returns number of columns
		'''	
		return len(self.df_columns) 
		
	def get_columns_of_current_data(self):
		'''
		Returns column names of current data
		'''
		return self.df_columns	
	
	def get_columns_of_df_by_id(self,id):
		'''
		'''
		return self.dfs[id].columns.values.tolist()
	
	def get_current_data_by_column_list(self,columnList, rowIdx = None, ignore_clipping = False):
		'''
		Returns sliced self.df
		row idx - boolean list/array like to slice data further.
		'''	
		return self.get_data_by_id(self.currentDataFile,False,rowIdx,ignore_clipping)[columnList]
		
		
	def get_current_data(self, rowIdx = None, ignoreClipping = False):
		'''
		Returns current df 
		'''
		return self.get_data_by_id(self.currentDataFile,False,rowIdx,ignoreClipping) 
		
		
	def get_data_by_id(self,id, setDataToCurrent = False, rowIdx = None, ignoreClipping = False):
		'''
		Returns df by id that was given in function: addDf(self)..
		'''

		if id in self.clippings:
			rowIdx = self.clippings[id]
		
		if setDataToCurrent:
		
			self.set_current_data_by_id(id = id) 
			
			if ignoreClipping or rowIdx is None:
				
				return self.df
			
			else: 
				return self.df.loc[rowIdx,:]
				
		else:
			if ignoreClipping or rowIdx is None:
			
				return self.dfs[id]
			else:
				return self.dfs[id].loc[rowIdx,:]
	
	def get_file_name_of_current_data(self):
		'''
		Returns the file name of currently selected data frame.
		'''
		return self.fileNameByID[self.currentDataFile]
			
	def get_groups_by_column_list(self,columnList, sort = False):
		'''
		Returns gorupby object of selected columnList
		'''
		
		if isinstance(columnList,list):
			groupByObject = self.df.groupby(columnList,sort = sort)
			
			return groupByObject
		else:
			return
		
	
	def get_id_of_current_data(self):
		'''
		Returns the currently active data frame ID.
		'''
		return self.currentDataFile
		
	
	def get_next_available_id(self):
		'''
		To provide consistent labeling, use this function to get the id the new df should be added
		'''
		self.dataFrameId += 1
		idForNextDataFrame = 'DataFrame: {}'.format(self.dataFrameId)
		
		return idForNextDataFrame
	
	def get_row_number(self):
		'''
		Returns the number of rows.
		'''
		return len(self.df.index)	
	
	def get_unique_values(self,categoricalColumn, forceListOutput = False):
		'''
		Return unique values of a categorical column. If multiple columns are
		provided in form of a list. It returns a list of pandas series having all
		unique values.
		'''
		if isinstance(categoricalColumn,list):
			if len(categoricalColumn) == 1:
				categoricalColumn = categoricalColumn[0]
				uniqueCategories = self.df[categoricalColumn].unique()
			else:
				collectUniqueSeries = []
				for category in categoricalColumn:
					collectUniqueSeries.append(self.df[category].unique())
				return collectUniqueSeries
		else:
			uniqueCategories = self.df[categoricalColumn].unique()
		if forceListOutput:
			return [uniqueCategories]
		else:
			return uniqueCategories



	def get_positive_subsets(self,numericColumn,categoricalColumns,inputData):
		'''
		'''
		data = pd.melt(inputData[numericColumn+categoricalColumns], 
									categoricalColumns, var_name = 'Columns', 
									value_name = 'Value')
		dataCombined = pd.DataFrame()
		complColumns = ['Complete']+categoricalColumns
		for category in complColumns:
			if category == 'Complete':
				dataCombined.loc[:,category] = data['Value']
			else:
				subset = data[data[category] == '+']['Value']
				dataCombined = pd.concat([dataCombined,subset],axis=1)
		dataCombined.columns = complColumns
		dataCombined.loc[:,'intIdxInstant'] = inputData.index	
		return dataCombined, complColumns	
							
							
							
					
						
	
	def iir_filter(self,columnNameList,n):
		'''
		Uses an iir filter to smooth data. Extremely useful in time series.
		'''
		b = [1.0/n]*n
		a=1
		newColumnNames = [self.evaluate_column_name('IIR(n_{})_{}'.format(n,columnName)) for columnName in columnNameList]
		
		transformedDataFrame = self.df[columnNameList].apply(lambda x: lfilter(b,a,x))

		self.df[newColumnNames] = transformedDataFrame
		self.update_columns_of_current_data()
		return newColumnNames
		
		
	
	def insert_data_frame_at_index(self, dataFrame, columnList, indexStart, indexList = None):
		'''
		Inserts Data Frame at certain location
		'''
		if any(columnName in self.df_columns for columnName in columnList):		
			return   
			
		for n,columnName in enumerate(columnList):
			
		
			if indexList is None:
				index = indexStart + 1 + n 
			else:
				index = indexList[n]
			newColumnData = dataFrame[columnName]
				
			self.insert_column_at_index(index,columnName,newColumnData)
		
			
			
	def insert_column_at_index(self, index, columnName, newColumnData):
		'''
		Inserts data at given index in current df.
		'''
		self.df.insert(index,columnName,newColumnData) 
		
	def join_series_to_currently_selected_df(self,series):
		'''
		'''
		self.df = self.df.join(series)
		self.update_columns_of_current_data()
		
	def join_df_to_currently_selected_df(self,dfToAdd, exportColumns = False):
		'''
		Joins another dataframe onto the currently selected one
		'''
		dfToAdd = self.evaluate_columnNames_of_df(dfToAdd)
		self.df = self.df.join(dfToAdd, rsuffix='_', lsuffix = '' ) 
		self.update_columns_of_current_data()
		
		if exportColumns:
			return dfToAdd.columns.values.tolist()
		
	def join_df_to_df_by_id(self,dfToAdd,id):
		'''
		Joins another data frame onto the df that is defined by id
		'''
		saveId = self.currentDataFile
		# we need to change to have a suitable evaluation
		self.set_current_data_by_id(id)
		#df = self.dfs[id]
		dfToAdd = self.evaluate_columnNames_of_df(dfToAdd)
		self.df = self.df.join(dfToAdd,rsuffix='_', lsuffix = ''  ) 
		self.save_current_data()
		self.update_columns_of_current_data()
		self.set_current_data_by_id(saveId)
				
		
	def join_missing_columns_to_other_df(self,otherDf, id, definedColumnsList = []):
		'''
		'''
		if id == self.currentDataFile:
			storedData = self.df
		else:
			storedData = self.dfs[id]
		columnsJoinDf = otherDf.columns
		
		if len(definedColumnsList) == 0:
			columnsMissing = [columnName for columnName in storedData.columns  if columnName\
			 not in columnsJoinDf]
		else:
			# check if data are not in tobe joined df but are in general in the df that the 
			# values are taken from
			columnsMissing = [columnName for columnName in definedColumnsList if columnName \
			not in columnsJoinDf and columnName in storedData.columns]
		
		if len(columnsMissing) != 0: 
			resultDataFrame = otherDf.join(storedData[columnsMissing])
			return resultDataFrame
		else:
			return otherDf
		
		
		
	def melt_data_by_column(self,columnNameList):
		'''
		Melts data frame.
		'''
		indxName = self.add_count_through_column(columnName = 'PriorMeltIndex')
		idVars = [column for column in self.df.columns if column not in columnNameList] #all columns but the selected ones
		valueName = self.evaluate_column_name('melt_value{}'.format(columnNameList).replace("'",''))
		variableName = self.evaluate_column_name('melt_variable{}'.format(columnNameList).replace("'",''))		
		
		meltedDataFrame = pd.melt(self.df, id_vars = idVars, value_vars = columnNameList,
								var_name = variableName,
								value_name = valueName)
		## determine file name
		baseFile = self.get_file_name_of_current_data()	
		numMeltedfiles = len([file for file in self.fileNameByID.values() if 'Melted_' in file])			
		fileName =  'Melted_{}_of_{}'.format(numMeltedfiles,baseFile)	
		
		#delete indexColumn again
		self.delete_columns_by_label_list([indxName])
		
		id = self.get_next_available_id()					
		self.add_data_frame(meltedDataFrame,id = id, fileName = fileName)
		
		return id,fileName,self.dfsDataTypesAndColumnNames[id]
		
	def unstack_column(self,columnName, separator = ';', verbose = False):
		'''
		Unstacks column. 
		'''
		if verbose:
			progressBar = Progressbar('Unstacking ..')
			progressBar.update_progressbar_and_label(4,'Staring')
		
		row_accumulator = []
		
		def splitListToRows(row, separator):
			split_row = row[columnName].split(separator)
			n = self.df.index.get_loc(row.name)
			if verbose and n % 50 == 0:
				progressBar.update_progressbar_and_label(n/nRows*100,'Working .. {}/{}'.format(n,nRows))
			for s in split_row:
				new_row = row.to_dict()
				new_row[columnName] = s
				row_accumulator.append(new_row)    
		
        
		nRows = len(self.df.index)
        
		self.df.apply(splitListToRows, axis=1, args = (separator, ))
		unstackedDf = pd.DataFrame(row_accumulator)
		
		#acquire name and source file
		
		baseFile = self.get_file_name_of_current_data()	
		fileName = 'Unstack_{}[{}]'.format(columnName,baseFile)
		id = self.get_next_available_id()	
		self.add_data_frame(unstackedDf,id = id, fileName = fileName)				
		if verbose:
		
			progressBar.update_progressbar_and_label(100,'Done ..')
			progressBar.close()
			
		return id,fileName,self.dfsDataTypesAndColumnNames[id]	
		
	
	def transform_data(self,columnNameList,transformation):
		'''
		Calculates data transformation and adds these to the data frame.
		'''	
		newColumnNames = [self.evaluate_column_name('{}_{}'.format(transformation,columnName)) \
		for columnName in columnNameList]
		
		if transformation == 'Z-Score_row':
			transformation = 'Z-Score'
			t1 = time.time()			
			axis_ = 1
		elif transformation == 'Z-Score_col':
			transformation = 'Z-Score' 
			axis_ = 0 
		else:
			axis_ = 0 
		
		if 'Z-Score' in transformation:
			transformedDataFrame = pd.DataFrame(scale(self.df[columnNameList].values, axis = axis_),
				columns = newColumnNames, index = self.df.index)
		else:
			transformedDataFrame = pd.DataFrame(
							calculations[transformation](self.df[columnNameList].values),
							columns = newColumnNames,
							index = self.df.index)
			
		if transformation != 'Z-Score':
			transformedDataFrame[~np.isfinite(transformedDataFrame)] = np.nan
		
		self.df[newColumnNames] = transformedDataFrame
		self.update_columns_of_current_data()
		
		return newColumnNames
	

	def update_data_frame(self,id,dataFrame):
		'''
		Updates dataframe, input: id and dataframe
		'''
		self.dfs[id] = dataFrame
		self.extract_data_type_of_columns(dataFrame,id)
		if id == self.currentDataFile:
			self.df = dataFrame
		
			

	def update_columns_of_current_data(self):
		'''
		Updates the variable: self.df_columns and renews the data type - column relationship.
		'''	
		self.df_columns = self.df.columns.values.tolist() 
		self.extract_data_type_of_columns(self.df,self.currentDataFile)	
		self.save_current_data()

	def replace_values_by_dict(self,replaceDict,id=None):
		'''
		Replaces values by dict. Dict must be nested in the form:
		{ColumnName:{Value:NewValue}}
		'''
		if id is None:
			pass
		else:
			self.set_current_data_by_id(id)
		self.df.replace(replaceDict,inplace=True)
		self.save_current_data()
		

	def resort_columns_in_current_data(self):
		'''
		Resorts columns alphabatically
		'''
		self.df.sort_index(axis = 1, inplace = True)
		self.update_columns_of_current_data()
		
		
	def remove_columns_with_low_variance(self,columns,thres = 0.5,copy=False):
		'''
		Calculates variance per columns. Remove columns bewlo threshold.
		'''
		data = self.df[columns]
		try:
			model = VarianceThreshold(thres).fit(data)
		except:
			return

		boolIndicator = model.get_support()
		newFeatures = list(compress(columns,boolIndicator))
		if np.sum(boolIndicator) == len(columns):
			return 'Same'
		
		if copy:
			newFeatureNames = [self.evaluate_column_name('VarFilt_{}'.format(feature)) \
						   for feature in newFeatures]
			self.df[newFeatureNames] = self.df[newFeatures]
			self.update_columns_of_current_data()	
				
			return newFeatureNames
		else:
			
			toDelete = [feature for n,feature in enumerate(columns) if boolIndicator[n] == False]
			self.delete_columns_by_label_list(toDelete)
			return toDelete
		
		
		
				
	def save_current_data(self):
		'''
		Save the current active df into the dictionary self.dfs
		'''

		self.dfs[self.currentDataFile] = self.df	
	
	def set_current_data_by_id(self,id = None):
		'''
		Change current data by ID
		'''
		if id is None:
			return
		if id not in self.dfs:
			return
		
		if self.currentDataFile != id:
			if self.df.empty == False:
		
				self.save_current_data() 
		
			self.df = self.dfs[id]
			self.df_columns = self.df.columns.values.tolist()
			self.currentDataFile = id		
				
	def sort_columns_alphabetically(self):
		'''
		Sorts collumns alphabetically
		'''
		
		self.df.reindex_axis(sorted(self.df_columns), axis=1, inplace=True)
		self.update_columns_of_current_data()
		
	def sort_columns_by_string_length(self, columnName):
		'''
		Sort columns by string length. 
		
		'''
		columnNameLen = 'stringLen{}'.format(columnName)
		internalColumnNameForSorting = self.evaluate_column_name(columnNameLen)
		if self.df[columnName].dtype in [np.int64,np.float64]:
			self.df[internalColumnNameForSorting] = self.df[columnName].astype('str').str.len()
		else:
			self.df[internalColumnNameForSorting] = self.df[columnName].str.len()
		
		self.df.sort_values(internalColumnNameForSorting,kind='mergesort',inplace=True) 
		
		self.delete_columns_by_label_list([internalColumnNameForSorting])
		
		
		
			
	def sort_columns_by_value(self,columnNameOrListOfColumns, kind = 'mergesort', ascending = True, na_position = 'last'):
		'''
		Sort rows in one or multiple columns.
		'''
		if isinstance(columnNameOrListOfColumns, str):
			columnNameOrListOfColumns = [columnNameOrListOfColumns]
		
		if self.currentDataFile in self.rememberSorting:
			
			columnNameThatWereSortedAscending = self.rememberSorting[self.currentDataFile]			
			changeToDescending = [col for col in columnNameOrListOfColumns if col in columnNameThatWereSortedAscending]
			## Check if all columns were sorted already in ascending order 
			numToDescending = len(changeToDescending)
			if numToDescending == len(columnNameOrListOfColumns) and numToDescending > 0:
				ascending = False
				columnNameThatWereSortedAscending = [col for col in columnNameThatWereSortedAscending if col not in columnNameOrListOfColumns] 
				self.rememberSorting[self.currentDataFile] = columnNameThatWereSortedAscending
	
		
		self.df.sort_values(by = columnNameOrListOfColumns, kind= kind,
							ascending = ascending,
							na_position = na_position,
							inplace = True)			
		if ascending:
			## save columns that were already sorted 
			if self.currentDataFile in self.rememberSorting:
			
				columnNameThatWereSortedAscending = self.rememberSorting[self.currentDataFile]
				columnNamesToAdd = [col for col in columnNameOrListOfColumns if col not in columnNameThatWereSortedAscending] 
				columnNameThatWereSortedAscending.extend(columnNamesToAdd)
				self.rememberSorting[self.currentDataFile] = columnNameThatWereSortedAscending
					
			else:
				self.rememberSorting[self.currentDataFile] = columnNameOrListOfColumns	
						
		return ascending		
		
	def split_columns_by_string(self,columnNameOrListOfColumns,splitString):
		'''
		Function to Split Columns by a String provided by the User.
		Columns that are not already an object, are first changed. This is more a precaution since there are only rare cases
		when this can happen if any. 
		Data are inserted at the correct position after the input column. 
		(Might be a bit slower than join function but usually does not handle many datacolumns)
		'''
		indexInTreeview = 'end'
		if isinstance(columnNameOrListOfColumns,str):
			columnNameOrListOfColumns = [columnNameOrListOfColumns]
		for columnName in columnNameOrListOfColumns:
			if self.df[columnName].dtype == 'object':
				df_split = self.df[columnName].str.split(splitString ,expand=True) 
			else:
				df_split= self.df[columnName].astype('str').str.split(splitString ,expand=True) 
			expandedSplitOnDataColumns = df_split.columns	
			if len(expandedSplitOnDataColumns) == 1:
				return None, None
				
			indexColumnInData = self.df_columns.index(columnName)
			indexInTreeview = self.dfsDataTypesAndColumnNames[self.currentDataFile]['object'].index(columnName)
			
			newColumnNames = df_split.columns = [self.evaluate_column_name("{}_[by_{}]_{}".format(columnName,splitString,colIndex), 
				useExact = True) for colIndex in expandedSplitOnDataColumns]
				
			df_split.fillna(self.replaceObjectNan,inplace=True)
			self.insert_data_frame_at_index(df_split,newColumnNames,indexColumnInData)
			
		self.update_columns_of_current_data()
		
		return newColumnNames, indexInTreeview 

			
	def rename_data_frame(self,id,fileName):
		'''
		'''
		self.fileNameByID[id] = fileName
						
			
	def rename_columnNames_in_current_data(self,replaceDict):
		'''
		Replaces column names in currently selected data frame.
		'''
		self.df.rename(str,columns=replaceDict,inplace=True)
		
		self.update_columns_of_current_data()		


	def metric_over_n_rows(self,columns,n,id = None, metric='mean'):
		
		if id is None:
			
			df = self.df[columns]
		groupedData = df.groupby(df.index // n * n)
		if hasattr(groupedData,metric):
			df = getattr(groupedData,metric)()
			return df 
		
		
		
			
	def shift_data_by_row_matches(self,id = None, matchColumn = '', 
								  adjustColumns = [], sort = False,
								  intervalData = {}, removeOtherData = False):
		'''
		Shift data
		'''
		if id is None:
			id = self.currentDataFile
		
		df = self.get_data_by_id(id, ignoreClipping = True)
		
		if matchColumn not in df.columns:
			return
			
		if any(col not in df.columns for col in adjustColumns):
			return
		
		if len(intervalData) == 0:
			return
			
		if sort:
			df.sort_values(matchColumn, inplace=True)
		
		if removeOtherData:
			columns = adjustColumns + [matchColumn]
			df = df[columns]
		
		collectDf = pd.DataFrame()
		timeCols = []
		columnsAdded = []
		intSteps = (df[matchColumn].max() - df[matchColumn].min())/len(df.index)
		
		for n,column in enumerate(adjustColumns):
			try:
				dfSub = pd.DataFrame()
				start, newZero = intervalData[column]
				endIdx = df.index[-1]
				startIdx = (df[matchColumn]-start).abs().argsort()[0]
				zeroIdx = (df[matchColumn]-newZero).abs().argsort()[0]
			
				timeCol = '{}{}'.format(matchColumn,n)
				dfSub[timeCol] = df[matchColumn].iloc[startIdx:endIdx,]
			
				timeCols.append(timeCol) 
			
				dfSub[column] = df[column].iloc[startIdx:endIdx,]
				dfSub.index = range(startIdx-zeroIdx,endIdx-zeroIdx)			
			

				collectDf = pd.concat([collectDf,dfSub],ignore_index=False, axis=1)
			except:
				pass
		indices = collectDf.index.tolist()
		minValue = indices[0] * intSteps
		
		maxValue = indices[-1] * intSteps
		
		collectDf['adj_interval'] = np.linspace(minValue,maxValue,num = len(collectDf.index))
		
		return collectDf			

		
		
			
		
				
		
	
		
		
		
	
	
	
	
	
	
		
		










