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

from math import isnan
from numba.np.ufunc import parallel
import numpy as np
from pandas.core.algorithms import isin
from scipy.signal import lfilter

from itertools import compress, chain, groupby
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import KernelDensity
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import scale
from threadpoolctl import threadpool_limits

from scipy.stats import pearsonr

from ..utils.stringOperations import getMessageProps, mergeListToString, findCommonStart, getRandomString
from ..filter.categoricalFilter import CategoricalFilter
from ..filter.numericalFilter import NumericFilter
from ..color.colorManager import ColorManager
from ..statistics.statistics import StatisticCenter
from ..normalization.normalizer import Normalizer
from ..transformations.transformer import Transformer
from ..proteomics.ICModifications import ICModPeptidePositionFinder
from .ICExcelExport import ICHClustExporter
from .ICSmartReplace import ICSmartReplace
from collections import OrderedDict
import itertools
#from modules.dialogs.categorical_filter import categoricalFilter
#from modules.utils import *

import numba as nb
from numba import prange, jit

from matplotlib.colors import to_hex
import time
import re 
import os 


def fasta_iter(fasta_name):
    """
    modified from Brent Pedersen
    Correct Way To Parse A Fasta File In Python
    given a fasta file. yield tuples of header, sequence
    """
    
    fh = open(fasta_name)
    faiter = (x[1] for x in itertools.groupby(fh, lambda line: line[0] == ">"))

    for header in faiter:
        # drop the ">"
        headerStr = header.__next__()[1:].strip()

        # join all sequence lines to one.
        seq = "".join(s.strip() for s in faiter.__next__())

        yield (headerStr, seq)



def z_score(x):
	mean = x.mean()
	std = x.std() 
	vector = (x-mean)/std
	return vector



def pearsonByRowsTwoArray(X,Y):
	idx = 0
	nRows = X.shape[0] * Y.shape[0]
	A = np.empty(shape=(nRows,2), dtype=np.float64)
	for n in range(X.shape[0]):
		for m in range(Y.shape[0]):
			y = Y[m,:]
			nonNaNIdx = [idx for idx in range(Y.shape[1]) if not np.isnan(X[n,idx]) and not np.isnan(Y[m,idx])]
			x = X[n,nonNaNIdx]
			y = Y[m,nonNaNIdx]
			if x.size > 2 and y.size > 2:
				r,p = pearsonr(x,y)
			
				A[idx,:] = [r,p]
				
			else:
				A[idx,:] = [np.nan,np.nan]
			idx+=1
	return A 

def pearsonByRowsTwoArrayN(X,Y):
	r = 0
	nRows = X.shape[0] * Y.shape[0]
	A = np.zeros(shape=(nRows,1))
	for n in range(X.shape[0]):
		x = X[n,:].flatten()
		for m in range(Y.shape[0]):
			y = Y[m,:].flatten()

			A[r] = _pearson(x,y)
			r+=1
	return A 


def corr2_coeff(A, B):
	# Rowwise mean of input arrays & subtract from input arrays themeselves
	nRows = A.shape[0] * B.shape[0]
	X = np.zeros(shape=(nRows,1))

	A_mA = A - np.nanmean(A, axis=1)[:, None]
	B_mB = B - np.nanmean(B,axis=1)[:, None]
	print(A_mA)
	
	#mulSum = np.nansum(mul, axis=1)
	# Sum of squares across rows
	ssA = np.nansum(A_mA**2,axis=1)
	ssB = np.nansum(B_mB**2, axis=1)
	r = 0
	for n in range(A.shape[0]):
		for m in range(B.shape[0]):

			xy = np.sum([x*y for x,y in zip(A_mA[n],B_mB[m]) if not np.isnan(x) and not np.isnan(y)])

			X[r] =   xy / np.sqrt(ssA[n] * ssB[m])
			r+=1
	return X



errorMessage = {"messageProps":{"title":"Error","message":"There was an error loading the file."}}
sepConverter = {"tab":"\t","space":"\s+"}
dTypeConv = {"float64":"Numeric Floats","int64":"Integers","object":"Categories"}
       
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

	def __init__(self, parent=None):
		self.parent = parent
		self.currentDataFile = None
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
		self.categoricalFilter = CategoricalFilter(self)
		self.numericFilter = NumericFilter(self)
		self.statCenter = StatisticCenter(self)
		self.colorManager = ColorManager(self)
		self.normalizer = Normalizer(self)
		self.transformer = Transformer(self)
		self.Plotter = None
	

	def setPlotter(self,plotter):
		""
		self.Plotter = plotter

	def addAnnotationColumnByIndex(self,dataID, indices, columnName):
		""
		if dataID in self.dfs:
			columnName = self.evaluateColumnName(columnName,dataID)
			annotationData = pd.Series(
					[self.getNaNString()] * self.dfs[dataID].index.size, 
					index = self.dfs[dataID].index,
					name = columnName)
			annotationData.loc[indices] = "+"
			df = self.dfs[dataID].join(annotationData)
			self.updateData(dataID,df)

			funcProps = getMessageProps("Column added","Column {} was added to data.".format(columnName))
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]

			return funcProps
		else:
			return errorMessage


	def addColumnData(self,dataID,columnName,columnData,rowIndex = None, evaluateName = True,**kwargs):
		'''
		Adds a new column to the data
		'''
		
		if evaluateName:
			columnName = self.evaluateColumnName(columnName,dataID,**kwargs)
		if rowIndex is None:
			self.dfs[dataID].loc[:,columnName] = columnData
		else:
			self.dfs[dataID].loc[rowIndex,columnName] = columnData
		self.extractDataTypeOfColumns(dataID)
		
		funcProps = getMessageProps("Column added","Column {} was added to data.".format(columnName))
		funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]

		return funcProps

	def addIndexColumn(self,dataID,*args,**kwargs):
		""
		if dataID in self.dfs:
			dfShape, rowIdx = self.getDataFrameShape(dataID)
			numRows, _ = dfShape
			columnName = "Index"
			idxData = pd.DataFrame(np.arange(numRows), index=rowIdx, columns=[columnName])
			return self.joinDataFrame(dataID,idxData)
		else:
			return errorMessage
	
	def addGroupIndexColumn(self,dataID,columnNames,*args,**kwargs):
		""
		if dataID in self.dfs:
			dfShape, rowIdx = self.getDataFrameShape(dataID)
			columnNames = columnNames.values.tolist()
			data = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
			dataGroupby = data.groupby(columnNames,sort=False).cumcount()
			newColumnName = "groupIndex:({})".format(mergeListToString(columnNames))
			idxData = pd.DataFrame(dataGroupby.values,index=dataGroupby.index, columns=[newColumnName])
			return self.joinDataFrame(dataID,idxData)
		

	def addDataFrameFromTxtFile(self,pathToFile,fileName,loadFileProps = None, returnPlainDf = False):
		"Load Data frame from txt file"
		try:
			if loadFileProps is None:
				loadFileProps = self.loadDefaultReadFileProps()
			loadFileProps = self.checkLoadProps(loadFileProps)
			df = pd.read_csv(pathToFile,**loadFileProps)
		except Exception as e:
			print(e)
			return errorMessage
		if returnPlainDf:
			return df
		else:
			return self.addDataFrame(df,fileName=fileName, cleanObjectColumns = True)

	def addDataFrameFromExcelFile(self,pathToFile,fileName,loadFileProps = None, returnPlainDf = False):
		try:
			if loadFileProps is not None and "sheet_name" in loadFileProps and loadFileProps["sheet_name"] is not None:
				loadFileProps["sheet_name"] = loadFileProps["sheet_name"].split(";")
			df = pd.read_excel(pathToFile,**loadFileProps)
		except Exception as e:
			return errorMessage
		if returnPlainDf:
			return df
		else:
			if isinstance(df,dict):#multiple sheets
				dataFrameNames = []
				for sheetName, sheetData in df.items():
					sheetFileName = "{}({})".format(fileName,sheetName)
					dataFrameNames.append(sheetFileName)
					rKwargs = self.addDataFrame(sheetData,fileName=sheetFileName, cleanObjectColumns = True)
				msgKwargs = getMessageProps("Done..","Sheets loaded: {}".format(dataFrameNames))
				rKwargs["messageProps"] = msgKwargs
				return rKwargs
			elif isinstance(df,pd.DataFrame):
				return self.addDataFrame(df,fileName=fileName, cleanObjectColumns = True)
	

	def addDataFrame(self,dataFrame, dataID = None, fileName = '', 
							cleanObjectColumns = False):
		'''
		Adds new dataFrame to collection.
		'''
		if dataID is None:
			dataID  = self.get_next_available_id()
		dataFrame = self.checkForInternallyUsedColumnNames(dataFrame)
		self.dfs[dataID] = dataFrame
		self.extractDataTypeOfColumns(dataID)
		self.rename_data_frame(dataID,fileName)

		rows,columns = self.dfs[dataID].shape

		#clean up nan in object columns
		if cleanObjectColumns:
			objectColumnList = self.dfsDataTypesAndColumnNames[dataID]["Categories"]
			self.fillNaInObjectColumns(dataID,objectColumnList)

		return {"messageProps":
				{"title":"Data Frame Loaded {}".format(fileName),
				"message":"{} loaded and added.\nShape (rows x columns) is {} x {}".format(dataID,rows,columns)},
				"columnNamesByType":self.dfsDataTypesAndColumnNames[dataID],
				"dfs":self.fileNameByID
			}
	
	def checkForInternallyUsedColumnNames(self,dataFrame):
		""
		FORBIDDEN_COLUMN_NAMES = ["color","size","idx","layer","None"]
		columnNamesToChange = [colName for colName in dataFrame.columns if colName in FORBIDDEN_COLUMN_NAMES]
		columnNamesNoChangeRequired = [colName for colName in dataFrame.columns if colName not in FORBIDDEN_COLUMN_NAMES]
		if len(columnNamesToChange) == 0:
			return dataFrame
		else:
			columnNameMapper = {} 
			for colName in columnNamesToChange:
				n = 0
				originalColName = str(colName)
				while colName in FORBIDDEN_COLUMN_NAMES or colName in columnNameMapper.values() or colName in columnNamesToChange or colName in columnNamesNoChangeRequired:
					
					colName = "{}_{}".format(originalColName,n)
					n += 1 

				columnNameMapper[originalColName] = colName

			dataFrame = dataFrame.rename(columns=columnNameMapper)
			return dataFrame

	def getQuickSelectData(self,dataID,filterProps):
		""
		if dataID in self.dfs:
			if all(filterProp in filterProps for filterProp in ["columnName","mode","sep"]):
				columnName = filterProps["columnName"]
				
				if filterProps["mode"] == "unique":
					
					sep = filterProps["sep"]
					#getUniqueCategroies returns a data frame, therefore index column 
					#to get a pandas Series (QuickSelect Model works with series)
					data = self.categoricalFilter.getUniqueCategories(dataID,columnName,splitString=sep)
					if isinstance(data,dict):
						return data
					elif isinstance(data,pd.DataFrame):
						data = data[columnName]
					else:
						return errorMessage	
					
				else:

					if columnName in self.dfs[dataID].columns:
						data = self.getDataByColumnNames(dataID,[columnName])["fnKwargs"]["data"][columnName]
					else:
						data = pd.Series()
				
				return {"messageProps":{"title":"Quick Select {}".format(columnName),
										"message":"Data were added to the Quick Select widget"},
						"data":data}
			
		return errorMessage

	def getDataFrameShape(self,dataID):
		""
		if dataID in self.dfs:
			if dataID in self.clippings:
				rowIdx = self.clippings[dataID]
				self.dfs.loc[rowIdx,:].shape, rowIdx
			else:
				return self.dfs[dataID].shape, self.dfs[dataID].index
		else:
			return (0,0)

	def groupbyAndAggregate(self,dataID,columnNames,groupbyColumn,metric="mean"):
		"""
		Aggregates data by a specific groupbyColumn. 
		ColumnNames can only be numeric.
		
		"""
		if dataID in self.dfs:
			requiredColumns = columnNames.append(pd.Series([groupbyColumn]),ignore_index=True)
			data = self.getDataByColumnNames(dataID,requiredColumns)["fnKwargs"]["data"]
			aggregatedData = data.groupby(by=groupbyColumn,sort=False).aggregate(metric)
			aggregatedData = aggregatedData.reset_index()
			return self.addDataFrame(aggregatedData,fileName = "{}(groupAggregate({}:{})".format(metric,self.getFileNameByID(dataID),groupbyColumn))
		else:
			return errorMessage

	def checkColumnNamesInDataByID(self,dataID,columnNames):
		""
		checkedColumnNames = []

		if isinstance(columnNames,str):
			columnNames = [columnNames]

		if not isinstance(columnNames,list):
			raise ValueError("Provide either list or string")
		
		if dataID in self.dfs:
			checkedColumnNames = [columnName for columnName in columnNames if columnName in self.dfs[dataID].columns] 
		
		return checkedColumnNames

	def loadDefaultReadFileProps(self):
		""
		config = self.parent.config 
		props = {
			"encoding":config.getParam("load.file.encoding"),
			"sep":config.getParam("load.file.column.separator"),
			"thousands":config.getParam("load.file.float.thousands"),
			"decimal": config.getParam("load.file.float.decimal"),
			"skiprows": config.getParam("load.file.skiprows"),
			"na_values":config.getParam("load.file.na.values")}
		return self.checkLoadProps(props)

	def checkLoadProps(self,loadFileProps):
		""
		if loadFileProps is None:
			return {"sep":"\t"}
		if loadFileProps["sep"] in ["tab","space"]:
			loadFileProps["sep"] = sepConverter[loadFileProps["sep"]]
		if "na_values" in loadFileProps and loadFileProps["na_values"] == "None":
			loadFileProps["na_values"] = None
		elif isinstance(loadFileProps["na_values"],str) and ";" in loadFileProps["na_values"]:
			loadFileProps["na_values"] = loadFileProps["na_values"].split(";")
		if "thousandas" not in loadFileProps or loadFileProps["thousands"] == "None":
			loadFileProps["thousands"] = None
		try:
			skiprows = int(loadFileProps["skiprows"])
		except:
			skiprows = 0
		loadFileProps["skiprows"] = skiprows

		return loadFileProps

	
	def columnRegExMatches(self,dataID,columnNames,searchString,splitString=";"):
		"""
		Return data indeces that match a regular expression. The regular expression
		is designed to match categories based on the provided serach string and split string
		A category  is a substring surrounded by split string. For example: 
		
		membrane;mitochondrion;nucleus

		as they appear in Gene Ontology expression. 

		Parameter 
		============

		Returns
		============
		
		"""

		regEx = self.categoricalFilter.buildRegex([searchString], withSeparator=True, splitString=splitString)
		data = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
		boolIdx = data[columnNames[0]].str.contains(regEx)

		return data.index[boolIdx]

	def copyDataFrameSelection(self,dataID,columnNames):
		""
		if dataID not in self.dfs:
			return getMessageProps("Error ..","DataID unknown ..")
		checkedColNames = self.checkColumnNamesInDataByID(dataID,columnNames)
		return self.copyDataFrameToClipboard(data = self.dfs[dataID].loc[:,checkedColNames])


	def joinAndCopyDataForQuickSelect(self,dataID,columnName,selectionData,splitString):
		""

		dataToConcat = []
		for checkedValue,checkedColor,userDefinedColors,checkSizes in selectionData.values:

			dataIdx = self.columnRegExMatches(dataID,[columnName],checkedValue,splitString)
			valueSubset = self.dfs[dataID].loc[dataIdx]
			valueSubset["QuickSelectValue"] = [checkedValue] * dataIdx.size
			
			if pd.isna(userDefinedColors):
				valueSubset["QuickSelectColor"] = [checkedColor] * dataIdx.size
			else:
				valueSubset["QuickSelectColor"] = [userDefinedColors] * dataIdx.size

			valueSubset["QuickSelectSize"] = [checkSizes] * dataIdx.size
			dataToConcat.append(valueSubset)

		data = pd.concat(dataToConcat,ignore_index=True)
		data.to_clipboard()
		return getMessageProps("Done ..","Quick Select data copied to clipboard. Data might contain duplicated rows.")

	def copyDataFrameToClipboard(self,dataID=None,data=None,attachDataToMain = None):
		""
		if dataID is None and data is None:
			return {"messageProps":{"title":"Error",
								"message":"Neither id nor data specified.."}
					}	
		elif dataID is not None and dataID in self.dfs and attachDataToMain is not None:
			#attach new data to exisisitng
			dataToCopy = attachDataToMain.join(self.dfs[dataID])
			dataToCopy.to_clipboard(excel=True)

		elif dataID is not None and dataID in self.dfs:
			data = self.getDataByDataID(dataID)
			data.to_clipboard(excel=True)

		elif isinstance(data,pd.DataFrame):

			data.to_clipboard(excel=True)
		
		else:
			return  {"messageProps":{"title":"Error",
								"message":"Unknown error.."}
					}	

		return  getMessageProps("Data copied","Selected data copied to clipboard..")
					


	def readDataFromClipboard(self):
		""
		try:
			
			data = pd.read_clipboard(**self.loadDefaultReadFileProps(), low_memory=False)
		except Exception as e:
			return getMessageProps("Error ..","There was an error loading the file from clipboard." + e)
		localTime = time.localtime()
		current_time = time.strftime("%H:%M:%S", localTime)
		return self.addDataFrame(data, fileName = "pastedData({})".format(current_time),cleanObjectColumns = True)

	def rowWiseCalculations(self,dataID,calculationProps,operation = "subtract"):
		""
		funcKeyMatch = {"Mean":np.nanmean,"Sum":np.nansum,"Median":np.nanmedian}
		operationKeys = {"subtract":np.subtract,"divide":np.divide,"multiply":np.multiply,"addition":np.add}
		data = self.dfs[dataID]
		collectResults = pd.DataFrame(index = data.index)
		for columnName, calcProp in calculationProps.items():
			X = data[columnName].values 
			if calcProp["metric"] == "Specific Column":
				Y = data[calcProp["columns"]].values 
				#ugly hack to not show the long 'Specific colum" in the name
				calcProp["metric"] = ""
			elif calcProp["metric"] in funcKeyMatch:
				Y = funcKeyMatch[calcProp["metric"]](data[calcProp["columns"]].values,axis=1)
			#perform calculation
			XYCalc = operationKeys[operation](X,Y)
			if isinstance(calcProp["columns"],str):
				newColumnName = "{}:({}):{}({})".format(columnName,operation,calcProp["metric"],calcProp["columns"])
			else:
				newColumnName = "{}:({}):{}({})".format(columnName,operation,calcProp["metric"],mergeListToString(calcProp["columns"].flatten(),","))
			collectResults[newColumnName] = XYCalc

		return self.joinDataFrame(dataID,collectResults)
	
	def sortColumns(self,dataID,sortedColumnDict):
		""
		sortedColumns = []
		for v in sortedColumnDict.values():
			sortedColumns.extend(v.values.flatten())
		self.dfs[dataID] = self.dfs[dataID][sortedColumns]
		return getMessageProps("Done..","Columns resorted")

	def dropColumns(self,dataID,columnNames):
		""
		if dataID in self.dfs:
			self.dfs[dataID] = self.dfs[dataID].drop(columns = columnNames, errors = "ignore")
			self.extractDataTypeOfColumns(dataID)
			taskCompleteKwargs = getMessageProps("Column (s) deleted","Selected columns deleted.")
			taskCompleteKwargs["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			taskCompleteKwargs["columnNames"] = columnNames
			return taskCompleteKwargs
		else:
			taskCompleteKwargs =  getMessageProps("Error ..","There was an error deleting selected columns.")
			taskCompleteKwargs["columnNames"] = []
			return 

				
	def deleteData(self,dataID):
		'''
		Deletes DataFile by id.
		'''	
		if dataID not in self.dfs:
			return getMessageProps("Not found.","Data ID not found")
		fileName = self.fileNameByID[dataID]

		del self.dfs[dataID]
		del self.fileNameByID[dataID]
		del self.dfsDataTypesAndColumnNames[dataID]

		taskCompleteKwargs = getMessageProps("Data deleted.","{} deleted.".format(fileName))

		if len(self.dfs) > 0:
			dataID = list(self.dfs.keys())[0]
			taskCompleteKwargs["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			taskCompleteKwargs["dfs"] = self.fileNameByID
			return taskCompleteKwargs
		else:
			taskCompleteKwargs["columnNamesByType"] = self.getEmptyDataType()
			taskCompleteKwargs["dfs"] = dict()
			return taskCompleteKwargs
		
	def getEmptyDataType(self):

		return {"Numeric Floats":pd.Series(),"Categories":pd.Series(),"Integers":pd.Series()}
		

	def evaluateColumnsForPlot(self,columnNames, dataID, dataType, numUniqueValues = None):
		""

		evaluatedColumns = {"Numeric Floats": [], "Categories": []}
		if dataType in ["Categories","Integers"]:
			numUniqueValues = self.getNumberUniqueValues(dataID,columnNames)

		if dataType in self.dfsDataTypesAndColumnNames[dataID]:
			dtypeColumns = self.dfsDataTypesAndColumnNames[dataID][dataType].values.tolist()
			for columnName in columnNames:
				if columnName in dtypeColumns:
					if dataType == "Integers":
						evaluatedColumns["Categories"].append(columnName)
					else:
						evaluatedColumns[dataType].append(columnName)
		#merge dicts to together
		funcProps = {**getMessageProps("Added ..","Column added to graph."),
					 **{"columnNamesByType":evaluatedColumns,"numUniqueValues":numUniqueValues}}
		
		return funcProps

	def evaluateColumnName(self,columnName, dataID = None, columnList = None, extraColumnList = [], maxLength = 80):
		'''
		Check if the column name already exists and how often. Adds a suffix.
		'''
		if not isinstance(extraColumnList,list):
			extraColumnList = []
		if columnList is None:
			if dataID is not None and dataID in self.dfs:
				columnList = self.dfs[dataID].columns.tolist() + extraColumnList
			else:
				return columnName
				
		if len(columnName) > maxLength-3:
			columnName = columnName[:maxLength-30]+'__'+columnName[-30:]

		count = 0
		evalColumnName = columnName
		while evalColumnName in columnList:
			if "_" in evalColumnName and evalColumnName[-2:].isdigit() and evalColumnName.split("_")[-1].isdigit():
				removeChar = len(evalColumnName.split("_")[-1])
				count += 1
				try:
					evalColumnName = evalColumnName[:-removeChar] + "{:02d}".format(int(float(evalColumnName.split("_")[-1])) + count)
				except Exception as e:
					evalColumnName = "{}_{:02d}".format(evalColumnName, count)
			else:
				evalColumnName = "{}_{:02d}".format(evalColumnName, count)
			
		return evalColumnName

	def evaluateColumMapper(self,dataID,columnNameMapper):
		""
		evalMapper = {}
		savedNewNames = []
		restColumnNames = [columnName for columnName in self.dfs[dataID] if columnName not in columnNameMapper]
		for oldName, newName in columnNameMapper.items():
			if oldName in self.dfs[dataID]:
				checkAgainstColumns = restColumnNames + savedNewNames
				evalColumnName = self.evaluateColumnName(newName, columnList = checkAgainstColumns)
				savedNewNames.append(evalColumnName)
				evalMapper[oldName] = evalColumnName
		return evalMapper
	

	def extractDataTypeOfColumns(self,dataID):
		'''
		Saves the columns name per data type. In InstantClue there is no difference between
		objects and others non float, int, bool like columns.
		'''
		dataTypeColumnRelationship = dict() 
		for dataType in ['float64','int64','object']:
			try:
				if dataType != 'object':
					dfWithSpecificDataType = self.dfs[dataID].select_dtypes(include=[dataType])
				else:
					dfWithSpecificDataType = self.dfs[dataID].select_dtypes(exclude=['float64','int64'])
			except ValueError:
				dfWithSpecificDataType = pd.DataFrame() 		
			columnHeaders = dfWithSpecificDataType.columns.values.tolist()
			dataTypeColumnRelationship[dTypeConv[dataType]] = pd.Series(columnHeaders)
				
		self.dfsDataTypesAndColumnNames[dataID] = dataTypeColumnRelationship	
	
	def exportHClustToExcel(self,dataID,pathToExcel,clusteredData,colorArray,totalRows,clusterLabels,clusterColors,quickSelectData):
		""

		dataColumns = self.getPlainColumnNames(dataID).values.tolist()
		clusterColumns = clusteredData.columns.values.tolist() 
		extraDataColumns = [columnName for columnName in dataColumns if columnName not in clusterColumns]
		columnHeaders = ["Cluster ID"] + clusterColumns + extraDataColumns
		
		rowIdx = clusteredData.index
		extraData = self.getDataByColumnNames(dataID,extraDataColumns,rowIdx=rowIdx)["fnKwargs"]["data"]
		if quickSelectData is not None:
			extraData["QuickSelect"] = np.full(extraData.index.size,"")
			extraData.loc[quickSelectData[0]["dataIndexInClust"],"QuickSelect"] = [to_hex(c) for c in quickSelectData[1]]
			columnHeaders.append("QuickSelect")
		exporter = ICHClustExporter(pathToExcel,clusteredData,columnHeaders,colorArray,totalRows,extraData,clusterLabels,clusterColors)
		exporter.export()
		
		return getMessageProps("Saved ..","Cluster map saved: {}".format(pathToExcel))

	def explodeDataByColumn(self,dataID,columnNames,splitString=None):
			"Splits rows by splitString into lists and then applies explode on that particular column"
			if dataID in self.dfs:
				columnName = columnNames.values[0]
				data = self.dfs[dataID].copy()
				if splitString is None:
					splitString = self.parent.config.getParam("explode.split.string")
				data[columnName] = data[columnName].str.split(splitString,expand=False)
				explodedData = data.explode(columnName, ignore_index=True)
				fileName = self.getFileNameByID(dataID)
				return self.addDataFrame(explodedData, fileName="explode({})::{}".format(fileName,columnName))

			else:
				return errorMessage

	def fillNaInObjectColumns(self,dataID,columnLabelList, naFill = None):
		'''
		Replaces nan in certain columns by value
		'''
		if dataID in self.dfs:
			if naFill is None:
				naFill = self.replaceObjectNan
			self.dfs[dataID][columnLabelList] = self.dfs[dataID][columnLabelList].fillna(naFill)

	
	def filterFastaFileByColumnIDs(self,dataID,columnNames,fastaFile):
		"Take a list of ids and match them to a fasta file. The new file is create in the original fasta."
		uniprotTargetList = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"].values[:,0]
		fastaBaseFile = os.path.basename(fastaFile)
		dirname = os.path.dirname(fastaFile)
		filteredFastFile = "{}_filtered({}).fasta".format(fastaBaseFile,columnNames.values[0])
		filteredFastaPath = os.path.join(dirname,filteredFastFile)
		regExpStr = self.parent.config.getParam("reg.exp.fasta.filter")
		escape = self.parent.config.getParam("reg.exp.escape")
		if escape:
			regExpStr = re.escape(regExpStr)
		nMatches  = 0 
		with open(filteredFastaPath,"w") as f:

			for headerStr,seq in fasta_iter(fastaFile):
				match = re.search(regExpStr, headerStr)
				uniprotID = match.group(1)
				
				if uniprotID in uniprotTargetList:
		
					f.write(">"+headerStr+"\n")
					f.write(seq+"\n")
					nMatches += 1


		return getMessageProps("Done..","Fasta file filtered and saved. {} id matched.\nThe path to the file: {}".format(nMatches,filteredFastaPath))


	def getGroupsbByColumnList(self,dataID, columnList, sort = False, as_index = True):
		'''
		Returns gorupby object of selected columnList
		'''
		if isinstance(columnList,pd.Series):
			columnList = columnList.values.tolist()
		
		if isinstance(columnList,list):
			groupByObject = self.dfs[dataID].groupby(columnList,sort = sort,as_index=as_index)
			return groupByObject

	def getColumnNamesByDataID(self,dataID):
		"Returns Dict of Column Names per DataFrame and Column Type (float,int,string)"
		
		if dataID in self.dfsDataTypesAndColumnNames:
			return {"messageProps":
					{"title":"Data Frame Updated",
					"message":"Data Frame Selection updated."},
					"columnNamesByType":self.dfsDataTypesAndColumnNames[dataID]}
					
		else:
			return errorMessage
	
	def getPlainColumnNames(self,dataID):
		""
		if dataID in self.dfs:
			return self.dfs[dataID].columns
		else:
			return []

	def getDataDescription(self,dataID,columnNames):
		""
		return self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"].describe()

	def getDataByColumnNames(self, dataID, columnNames, rowIdx = None, ignore_clipping = False):
		'''
		Returns sliced self.df
		row idx - boolean list/array like to slice data further.
		'''
		if isinstance(columnNames,pd.Series):
			columnNames = columnNames.values.tolist()
	
		fnComplete = {"fnName":"set_data","fnKwargs":{"data":self.getDataByDataID(dataID,rowIdx,ignore_clipping)[columnNames]}}
		return fnComplete

	def getDataByColumnNamesAndPlot(self, dataID, columnNames, activePlotter, rowIdx = None, ignore_clipping = False):
		'''
		Returns sliced self.df
		row idx - boolean list/array like to slice data further.
		'''
		fnComplete = self.getDataByColumnNames(dataID,columnNames,rowIdx,ignore_clipping)
		#activePlotter = self.Plotter.get_active_helper()
		if activePlotter is not None and hasattr(activePlotter,fnComplete["fnName"]):
			try:
				getattr(activePlotter,fnComplete["fnName"])(**fnComplete["fnKwargs"])
			except Exception as e:
				print(e)
		return {"newPlot":True,"fnName":"outsideThreadPlotting","fnKwargs":{}}


	def getDataByDataID(self,dataID, rowIdx = None, ignoreClipping = False):
		'''
		Returns df by id that was given in function: addDf(self)..
		'''

		if dataID not in self.dfs:
			return pd.DataFrame()

		if dataID in self.clippings:
			rowIdx = self.clippings[dataID]
		
		if ignoreClipping or rowIdx is None:
		
			return self.dfs[dataID]
		else:
			return self.dfs[dataID].loc[rowIdx,:]

	def getDataByColumnNameForWebApp(self,dataID,columnName):
		""

		data = self.getDataByColumnNames(dataID,[columnName])["fnKwargs"]["data"]
		data = data.rename(columns={columnName:"text"})
		data["idx"] = data.index
		return data.to_json(orient="records")

	def getNaNString(self):
		""
		return self.replaceObjectNan

	def getNumberUniqueValues(self,dataID,columnNames):
		"""
		"""
		resultDict = OrderedDict()
		if dataID in self.dfs:
			
			for categoricalColumn in columnNames:
				resultDict[categoricalColumn] = self.dfs[dataID][categoricalColumn].unique().size
		
		return resultDict

	def getUniqueValues(self, dataID, categoricalColumn, forceListOutput = False):
		'''
		Return unique values of a categorical column. If multiple columns are
		provided in form of a list. It returns a list of pandas series having all
		unique values.
		'''
		if isinstance(categoricalColumn,list):
			if len(categoricalColumn) == 1:
				categoricalColumn = categoricalColumn[0]
				uniqueCategories = self.dfs[dataID][categoricalColumn].unique()
			else:
				collectUniqueSeries = []
				for category in categoricalColumn:
					collectUniqueSeries.append(self.dfs[dataID][category].unique())
				return collectUniqueSeries
		else:
			uniqueCategories = self.dfs[dataID][categoricalColumn].unique()

		if forceListOutput:
			return [uniqueCategories]
		else:
			return uniqueCategories

	def hasData(self):
		""
		return len(self.dfs) > 0

	def hasTwoDataSets(self):
		""
		return len(self.dfs) > 1

	def renameColumns(self,dataID,columnNameMapper):
		""
		if dataID in self.dfs:
			columnNameMapper = self.evaluateColumMapper(dataID,columnNameMapper)
			self.dfs[dataID] = self.dfs[dataID].rename(mapper = columnNameMapper, axis = 1)
			#update columns names
			self.extractDataTypeOfColumns(dataID)
			funcProps = getMessageProps("Column renamed.","Column evaluated and renamed.")
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			funcProps["columnNameMapper"] = columnNameMapper
			return funcProps
		return getMessageProps("Error..","DataID not found.")

	def setClippingByFilter(self,dataID,columnName,filterProps, checkedLabels, checkedDataIndex = None):
		""
		#ccheck if data id exists
		if dataID in self.dfs:
			searchStrings = checkedLabels.astype(str).values.tolist()
			if isinstance(searchStrings,list) and len(searchStrings) == 0:
				funcProps = self.resetClipping(dataID)
				funcProps["maskIndex"] = None
				return funcProps
			else:
				if filterProps["mode"] == "unique":
					rowIdxBool = self.categoricalFilter.searchCategory(dataID,
																columnName,
																searchStrings,
																filterProps["sep"] if "sep" in filterProps else None)
				else:
					if self.dfs[dataID].index.size == checkedDataIndex.index.size:
						rowIdxBool = checkedDataIndex
					else:
						return getMessageProps("Error ..","Checked labels does not fit in length with data.")

				self.setClipping(dataID,rowIdxBool)
				funcProps = getMessageProps("Masking done.",
										    "Mask established graph will be updated.")
				funcProps["maskIndex"] = rowIdxBool
				return funcProps
		
		return getMessageProps("Error..","There was an error when clipping mask was established.")

		
	def evaluateColumnNameOfDf(self, df, dataID):
		'''
		Checks each column name individually to avoid same naming and overriding.
		'''
		columns = df.columns.values.tolist() 
		evalColumns = []
		#try:
		for columnName in columns:
			evalColumnName = self.evaluateColumnName(columnName,dataID=dataID,extraColumnList=evalColumns)
			evalColumns.append(evalColumnName)
		df.columns = evalColumns
		return df
		#except Exception as e:
		#	print(e)


	def getColorDictsByFilter(self,dataID,columnName,filterProps, checkedLabels, checkedDataIndex = None, checkedSizes = None, userColors = None, useBlit = False):
		"Color Data by Using the Quick Select Widget"
		if dataID in self.dfs:
				
				self.resetClipping(dataID)
				
				colorData, quickSelectColor, idxByCheckedLabel = self.colorManager.colorDataByMatch(dataID,columnName,
														colorMapName = "quickSelectColorMap",
														checkedLabels = checkedLabels,
														checkedDataIndex  = checkedDataIndex,
														checkedSizes = checkedSizes,
														userColors = userColors,
														splitString = filterProps["sep"] if "sep" in filterProps else None)
				
				funcProps =  {}#getMessageProps("Updated","Quick Selection Color Data Updated.")
				funcProps["propsData"] = colorData
				if filterProps["mode"] == "raw":
					
					funcProps["checkedColors"] = colorData["color"]
				
					df = pd.DataFrame(checkedLabels).join(colorData[["color","size"]])
				
					title = checkedLabels.name
					df.columns = ["group","color","size"]
					df = df[["color","size","group"]] #order is important here.
					df["internalID"] = [getRandomString() for n in df.index]
					
					funcProps["quickSelectData"] = df
					funcProps["title"] = title 

					funcProps["categoryIndexMatch"] = OrderedDict([(intID,[idx]) for  idx, intID in zip(checkedLabels.index,df["internalID"].values)])
					
				
				elif filterProps["mode"] == "unique":
					
					checkedLabels.name = "group"
					#quickSelectColor.name = "color"
					
					df = pd.DataFrame(checkedLabels)#.join(quickSelectColor)
					df["size"] = [colorData.loc[idx,"size"].values[0] for group,idx in idxByCheckedLabel.items()]
					df["color"] = [colorData.loc[idx,"color"].values[0] for group,idx in idxByCheckedLabel.items()]
					#df.columns = ["group","color"]
					df = df[["color","size","group"]]
					
					df["internalID"] = [getRandomString() for n in df.index]
					
					funcProps["quickSelectData"] = df
					funcProps["title"] = title = checkedLabels.name
					funcProps["checkedColors"] = quickSelectColor
					funcProps["categoryIndexMatch"] = OrderedDict([(intID,np.array(idxByCheckedLabel[checkedLabel])) for  checkedLabel, intID in df[["group","internalID"]].values])
					
				elif quickSelectColor is not None:
					funcProps["checkedColors"] = quickSelectColor

				funcProps["categoryEncoded"] = "QuickSelect"
				funcProps["ommitRedraw"] = useBlit
				
				
				return funcProps
		else:
			return errorMessage

	def getDataValue(self,dataID,columnName,dataIndex,splitString = None):
		""

		if dataID in self.dfs and columnName in self.dfs[dataID].columns:
			if not isinstance(dataIndex,pd.Series):
				dataIndex = pd.Series(dataIndex)
			values = self.dfs[dataID].loc[dataIndex,columnName].values.flatten()
			if splitString is not None:
				return itertools.chain.from_iterable([str(v).split(splitString) for v in values])
			else:
				return values
	
	def _getCategoricalData(self,dataID,ignoreClipping=True):
		""
		if dataID in self.dfs:
			categoricalColumns = self.getCategoricalColumns(dataID)
			data = self.getDataByColumnNames(dataID,categoricalColumns,ignore_clipping=ignoreClipping)["fnKwargs"]["data"]
			return data
		else:
			return errorMessage

	# def getCategoricalColorMap(self, dataID, columnNames):
	# 	""
	# 	colorProps = self.colorManager.getCategoricalColorMap(dataID,categoricalColumn)
	# 	funcProps = getMessageProps("Updated","Categorical column used for coloring")
	# 	funcProps["fnName"] = "change_color_by_categorical_columns"
	# 	funcProps["fnKwargs"] = colorProps 
	# 	return funcProps 

	def joinDataFrame(self,dataID,dataFrame):
		""
		if dataID in self.dfs:
			dataFrame = self.evaluateColumnNameOfDf(dataFrame,dataID)
			self.dfs[dataID] = self.dfs[dataID].join(dataFrame,rsuffix="_",lsuffix="")
			self.extractDataTypeOfColumns(dataID)
			objectColumnList = self.dfsDataTypesAndColumnNames[dataID]["Categories"]
			self.fillNaInObjectColumns(dataID,objectColumnList)
			funcProps = getMessageProps("Done ..","Data Frame has been added to {}.".format(dataID))
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			funcProps["dataID"] = dataID
			return funcProps
	
	def joinColumnToData(self,dataFrame,dataID,columnName):
		"Plain return"
		print(dataFrame,dataID,columnName)
		if dataID in self.dfs and columnName not in dataFrame.columns and columnName in self.dfs[dataID].columns:
			columnData = self.dfs[dataID][columnName]
			return dataFrame.join(columnData)


	def setClipping(self, dataID, rowIdxBool):
		'''
		data ID - ID that was given data frame when added to this class
		rowIdx - rows to be temporariley kept.
		'''
		self.clippings[dataID] = rowIdxBool	

	#def unstack
	def resetClipping(self,dataID):
		""
		if dataID in self.clippings:
			del self.clippings[dataID]
			return getMessageProps("Clipping reset.",
										"Clipping was removed. No Selection in Quick Select.")
		else:
			return getMessageProps("Error..","No cliiping mask found.")

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
	
		
	def getKernelDensity(self, dataID, columnNames, kernel = "gaussian", bandwidth = None, justData = False):
		'''
		'''

		data = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"].dropna() 
		if data.index.size > 2:
			if bandwidth is None:
				bandwidth = data.index.size**(-1/(data.columns.size+4)) 
			kde = KernelDensity(bandwidth=bandwidth,
                        kernel=kernel, algorithm='ball_tree')
			kde.fit(data) 
			kdeData = np.exp(kde.score_samples(data))
			if justData:
				return kdeData, data.index
			columnName = "kde::({})".format(mergeListToString(columnNames.values))			
			return self.addColumnData(dataID, columnName=columnName, columnData=kdeData, rowIndex = data.index)
		else:
			return getMessageProps("Error..","Not enough data rows after NaN filtering.")


	def getKernelDensityFromDf(self,data,kernel = "gaussian", bandwidth = None):
		""
		if data.index.size > 1:
			if bandwidth is None:
				bandwidth = data.index.size**(-1/(data.columns.size+4)) 
			kde = KernelDensity(bandwidth=bandwidth,
                        kernel=kernel, algorithm='ball_tree')
			kde.fit(data) 
			kdeData = np.exp(kde.score_samples(data))
			
			return kdeData, data.index
		return None, None
		
	def replaceColumns(self,dataID,columnNames,data):
		""
		if dataID in self.dfs:
			# if replacedColumnNames is not None and columnNames.size != replacedColumnNames.size:
			# 	return getMessageProps("Erorr..","There was an error in replacing data in place (incorrect column name number)")
			# elif replacedColumnNames is None:
			# 	replacedColumnNames = columnNames

			if self.dfs[dataID][columnNames].shape != data.shape:
				return getMessageProps("Error","Data that should be used for replacing are not of correct shape. {} vs {}".format(self.dfs[dataID][columnNames].shape,data.shape))
			
			else:
				#replace column names 
				#replaceColumns = dict([(k,replacedColumnNames.values[n]) for n,k in enumerate(columnNames.values)])
				#self.dfs[dataID] = self.dfs[dataID].rename(columns = replaceColumns, axis="columns")
				self.dfs[dataID][columnNames] = data 
				funcProps = getMessageProps("Done ..","Data have been updated.")
				funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
				funcProps["dataID"] = dataID
				return funcProps
		else:
			return errorMessage
				
			
         
	def replaceInColumns(self,dataID,findStrings,replaceStrings, specificColumns = None, dataType = "Numeric Floats", mustMatchCompleteCell = False):
		""
		if dataID in self.dfs:
			if specificColumns is None: #replace in columnHeaders
				newColumnNames  = None
				savedColumns = self.dfs[dataID].columns.values.tolist() 
				for n,fS in enumerate(findStrings):
					rS = replaceStrings[n] if len(replaceStrings) > 1 else replaceStrings[0]
					if newColumnNames is None:
						newColumnNames = self.dfs[dataID].columns
					if not mustMatchCompleteCell:
						newColumnNames = newColumnNames.str.replace(fS,rS,case=True,regex = False)
					else:
						newColumnNames = newColumnNames.replace(fS,rS)

				if np.unique(newColumnNames).size != newColumnNames.size:
					return getMessageProps("Error..","Replacing caused at least two column names to have the same name.")
				self.dfs[dataID].columns = newColumnNames
				self.extractDataTypeOfColumns(dataID)
				funcProps = getMessageProps("Column names replaced.","Column evaluated and replaced.")
				funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
				funcProps["dataID"] = dataID
				funcProps["columnNameMapper"] = dict([(oldColumnName,newColumnNames[n]) for n,oldColumnName in enumerate(savedColumns) if oldColumnName != newColumnNames[n]])
				return funcProps

			elif isinstance(specificColumns,list) and len(specificColumns) > 0 and all(x in self.getPlainColumnNames(dataID) for x in specificColumns):
				try:
				
					if dataType == "Numeric Floats":
						replaceValues = [float(x) if x != "nan" else np.nan for x in replaceStrings]#replace nan string with numpys nan
						findValues = [float(x) if x != "nan" else np.nan for x in findStrings]
					elif dataType == "Integers":
						replaceValues = [int(x) if x != "nan" else np.nan for x in replaceStrings]#replace nan string with numpys nan
						findValues = [int(x) if x != "nan" else np.nan for x in findStrings]
					else:
						replaceValues = [x if x != "nan" else self.replaceObjectNan for x in replaceStrings]#replace nan object string
						findValues = [str(x) if x != "nan" else self.replaceObjectNan for x in findStrings]
				except Exception as e:
					print(e)
					return getMessageProps("Error..","Strings could not be converted to required data type. Use 'nan' for NaN!")

				if len(findValues) > 1 and len(replaceValues) == 1:
					replaceValues = replaceValues * len(findValues)	
				if dataType not in ["Numeric Floats","Integers"] and not mustMatchCompleteCell:
					#convert to pandas series and ensure str data type
					for fS,rS in zip(findStrings,replaceStrings):
						#replace happens after each other
						self.dfs[dataID][specificColumns] = self.dfs[dataID][specificColumns[0]].astype(str).str.replace(fS, rS,regex=False,case=True)
				else:
					self.dfs[dataID][specificColumns] = self.dfs[dataID][specificColumns].replace(findValues, replaceValues)
				return getMessageProps("Column renamed.","Column evaluated and renamed.")

		else:
			return errorMessage
	


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
		
	
	def changeDataType(self, dataID, columnNames, newDataType):
		'''
		Changes the DataType of a List of column Names
		'''
		if dataID in self.dfs:
			if isinstance(columnNames,pd.Series):
				columnNames = columnNames.values

			try:
				self.dfs[dataID][columnNames] = self.dfs[dataID][columnNames].astype(newDataType)
			except:
				return getMessageProps("Error..","Changing data type failed.")
			
			if newDataType in ['object','str']:
				self.dfs[dataID][columnNames].fillna(self.replaceObjectNan,inplace=True)
			#update columns
			self.extractDataTypeOfColumns(dataID)
			funcProps = getMessageProps("Data Type changed.","Columns evaluated and data type changed.")
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			return funcProps
		else:
			return errorMessage

	
	def combineColumns(self,dataID,columnNames, sep='_'):
		'''
		'''
		if dataID in self.dfs:
			#attach columsn to merge and transform to string
			combinedRowEntries = []

			for columnName in columnNames.values[1:]:
				if columnName in self.dfs[dataID].columns:
					combinedRowEntries.append(self.dfs[dataID][columnName].astype(str))
			combineColumnName = 'combined_({})'.format(mergeListToString(columnNames))

			columnData = self.dfs[dataID][columnNames.values[0]].astype(str).str.cat(combinedRowEntries,sep=sep)

			return self.addColumnData(dataID,combineColumnName,columnData)

		else:
			return errorMessage 


	def duplicateColumns(self,dataID,columnNames):
		'''
		'''
		if dataID in self.dfs:
			#attach columsn to merge and transform to string
			newColumnNames = ["c:({})".format(colName) for colName in columnNames.values]
			data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
			copiedData = pd.DataFrame(data.values,index=data.index,columns=newColumnNames)
			return self.joinDataFrame(dataID,copiedData) 
		else:
			return errorMessage 


		
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
	
	
	def subset_data_on_category(self,columnNameList,id=None):
		'''
		'''
		if id is None:
			id = self.currentDataFile
			
		if all(x in self.dfs[id].columns for x in columnNameList):
			dataDict = OrderedDict()
			for columnName in columnNameList:
				uniqueValues = self.dfs[id][columnName].unique() 
				for uniqV in uniqueValues:
					if uniqV != "-":
						dataDict['{}({})'.format(uniqV,columnName)] = self.dfs[id][self.dfs[id][columnName] == uniqV]
			return dataDict	
		
				

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
		
		
	def countNaN(self,dataID, columnNames, grouping = None):
		""
		if dataID in self.dfs:
			if grouping is None:
				data = self.dfs[dataID][columnNames].isnull().sum(axis=1)
				return self.addColumnData(dataID,"count(nan):{}".format(mergeListToString(columnNames)),data)
			else:
				grouping = self.parent.grouping.getCurrentGrouping() 
				groupingName = self.parent.grouping.getCurrentGroupingName()
				columnNames = self.parent.grouping.getColumnNames(groupingName)
				X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
				countData = pd.DataFrame(index=X.index, columns = ["count(nan):{}".format(groupName) for groupName in grouping.keys()])
				for groupName, columnNames in grouping.items():
					
					countData["count(nan):{}".format(groupName)] = X[columnNames].isnull().sum(axis=1)
				
				return self.joinDataFrame(dataID,countData)

		else:
			return getMessageProps("Error ..","DataID not found.")



	def removeNaN(self, dataID, columnNames, how = "any", thresh = None, axis=0,*args,**kwargs):
		""
		if dataID in self.dfs:
			if axis == 0:
				if how in ["all","any"]:
					dataRows = self.dfs[dataID].index.size
					if thresh is not None:

						if thresh < 1:
							thresh = int(columnNames.size * thresh)
						else:
							thresh = int(thresh)

						self.dfs[dataID].dropna(subset = columnNames, thresh = thresh, inplace = True)

					else:
						
						self.dfs[dataID].dropna(subset = columnNames, how = how, inplace = True)
					#get number of removed NaNs
					nRemovedRows = dataRows - self.dfs[dataID].index.size 
					return getMessageProps("Removed ..","NaN were removed from data.\nIn total: {}".format(nRemovedRows))

				return getMessageProps("Error ..","No useful value for attribute 'how'.")
			
			else:

				cleanedDf = self.dfs[dataID][columnNames].dropna(how=how,axis="columns")
				removedColumns = columnNames.loc[columnNames.isin(cleanedDf.columns.values)]
				columnNames = [colName for colName in self.getPlainColumnNames(dataID) if colName not in columnNames.values] + cleanedDf.columns.values.tolist()
				self.dfs[dataID] = self.dfs[dataID][columnNames]
				#update columns names
				self.extractDataTypeOfColumns(dataID)
				funcProps = getMessageProps("Columns removed.","Column evaluated and removed.")
				funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
				funcProps["columnNames"] = removedColumns
				return funcProps
		else:
			return errorMessage




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
			columnName = columnName[:maxLength-30]+'__'+columnName[-30:]
			
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
		for dataType in ['float64','int64','object']:
			try:
				if dataType != 'object':
					dfWithSpecificDataType = dataFrame.select_dtypes(include=[dataType])
				else:
					dfWithSpecificDataType = dataFrame.select_dtypes(exclude=['float64','int64'])
			except ValueError:
				dfWithSpecificDataType = pd.DataFrame() 		
			columnHeaders = dfWithSpecificDataType.columns.values.tolist()
			dataTypeColumnRelationship[dTypeConv[dataType]] = pd.Series(columnHeaders)
				
		self.dfsDataTypesAndColumnNames[id] = dataTypeColumnRelationship
		
	
		
	def exportData(self,dataID,path = 'exportData.txt', columnOrder = None,fileFormat = "txt"):
		""
		if dataID in self.dfs:
			#remove deleted columns/save as resorted df
			if columnOrder is not None:
				newColumns = []
				for _ , columns in columnOrder.items():
					newColumns.extend(columns.values)
				self.dfs[dataID] = self.dfs[dataID][newColumns] 
			if fileFormat == "txt":
				self.dfs[dataID].to_csv(path, index=None, na_rep ='NaN', sep='\t')
			elif fileFormat == "xlsx":
				self.dfs[dataID].to_excel(path, index=False, sheet_name = "ICExport")
			elif fileFormat == "json":
				self.dfs[dataID].to_json(path, orient = self.parent.config.getParam("json.export.orient"), indent = 2)
			elif fileFormat == "md":
				self.dfs[dataID].to_markdown(path, tablefmt="grid")
			else:
				return getMessageProps("Error ..", "The used fileFormat unknown.")
			return getMessageProps("Exported.","Data exported to path:\n{}".format(path))
		else:
			return errorMessage
		

	def factorizeColumns(self,dataID, columnNames):
		'''
		Factorizes categories in columns. 
		'''
		if dataID in self.dfs:
			#remove deleted columns/save as resorted df
			newColumnNames = ["fac:({})".format(columnName) for columnName in columnNames.values]
			rawData = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
			idx = rawData.index
			facData = pd.DataFrame(index=idx,columns = newColumnNames )
			for n,columnName in enumerate(columnNames.values):
				facCodes, _ = pd.factorize(rawData[columnName].values)
				facData.loc[idx,newColumnNames[n]] = facCodes
			return self.joinDataFrame(dataID,facData)
		else:
			return errorMessage

	def filterDataByVariance(self,dataID,columnNames,varThresh=0.0,direction = "row"):
		""
		X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=False)["fnKwargs"]["data"]
		if direction == "row":
			varThresh = VarianceThreshold(threshold=varThresh).fit(X.values.T)
			support = varThresh.get_support()
			df = self.getDataByDataID(dataID).loc[support,:]
			return self.addDataFrame(df,fileName="varRowThresh:({})".format(self.getFileNameByID(dataID)))
		else:
			varThresh = VarianceThreshold(threshold=varThresh).fit(X.values)
			support = varThresh.get_support()
			df = self.getDataByDataID(dataID).loc[:,support]
			return self.addDataFrame(df,fileName="varColThresh:({})".format(self.getFileNameByID(dataID)))

	def fillNaNByGroupMean(self,dataID,columnNames = None):
		""
		if not self.parent.grouping.groupingExists():
			return getMessageProps("Error","No grouping found.")
		grouping = self.parent.grouping.getCurrentGrouping() 
		groupingName = self.parent.grouping.getCurrentGroupingName()
		columnNames = self.parent.grouping.getColumnNames(groupingName)
		X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
		replacedData = pd.DataFrame(index=X.index, columns = ["nanByGroupMean:{}".format(colName) for colName in X.columns])
		for groupName, groupColumns in grouping.items():
			groupColumnsForReplacedData = ["nanByGroupMean:{}".format(colName) for colName in groupColumns]
			groupData = X[groupColumns]
			#boolIdx = groupData.isna().sum(axis=1) > 1

			arr = groupData.values
			nanMean = np.nanmean(arr,axis=1, keepdims = True)
			#nanMean[boolIdx.values] = np.nan
			replacedData.loc[X.index,groupColumnsForReplacedData] = np.where(np.isnan(arr),nanMean,arr)
		
		return self.joinDataFrame(dataID,replacedData.astype(float))


	def fillNaNBySmartReplace(self,dataID,columnNames,grouping,**kwargs):
		""
		if dataID in self.dfs:
			try:
				smartRrep = ICSmartReplace(grouping=grouping,**kwargs)
				X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
				X = smartRrep.fitTransform(X)
				#print(X)
				self.dfs[dataID].loc[X.index,X.columns] = X
				return getMessageProps("Done ..","Replacement done.")

			except Exception as e:
				print(e)
		else:
			return errorMessage


	def summarizeGroups(self,dataID,grouping,metric,**kwargs):
		""
		if dataID in self.dfs:
			for groupName, columnNames in grouping.items():

				data = self.transformer.summarizeTransformation(dataID,columnNames,metric=metric,justValues = True)
				columnName = "s:{}({})".format(metric,groupName)
				self.addColumnData(dataID,columnName,data)

			completeKwargs = getMessageProps("Done..","Groups summarized. Columns added.")
			completeKwargs["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			return completeKwargs
		else:
			return errorMessage

	def removeDuplicates(self,dataID,columnNames):
		""
		if dataID in self.dfs:
			df = self.dfs[dataID].drop_duplicates(subset=columnNames.values)
			fileName = self.getFileNameByID(dataID)
			return self.addDataFrame(df,fileName="dropDup:({})".format(fileName))
		else:
			return errorMessage

	def replaceSelectionOutlierWithNaN(self,dataID,columnNames):
		""
		if dataID in self.dfs:
			try:
				m = self.parent.config.getParam("outlier.iqr.multiply")
				copy = self.parent.config.getParam("outlier.copy.results")
				if copy:
					cleanColumns = ["reO:{}".format(colName) for colName in columnNames.values]
					cleanData = pd.DataFrame(np.zeros(shape=(self.dfs[dataID].index.size,columnNames.size)), 
											index=self.dfs[dataID].index, 
											columns =  cleanColumns)
				
				X = self.dfs[dataID][columnNames].values
				Q = np.nanquantile(X,q=[0.25,0.5,0.75],axis=1)
				Q1 = Q[0,:].reshape(X.shape[0],1)
				M = Q[1,:].reshape(X.shape[0],1)
				Q3 = Q[2,:].reshape(X.shape[0],1)
				IQR = np.abs(Q3-Q1)

				outlierBool = np.logical_or(X > M + m * IQR, X < M - m * IQR)
				X[outlierBool] = np.nan
				if copy:
					cleanData[cleanColumns] = X 
					return self.joinDataFrame(dataID,cleanData)
				else:
					self.dfs[dataID][columnNames] =  X
					return getMessageProps("Done ..","Outlier replaced with NaN. Dataframe updated.")

			except Exception as e:
				print(e)

	def replaceGroupOutlierWithNaN(self,dataID,grouping):
		""
		if dataID in self.dfs:
			
				m = self.parent.config.getParam("outlier.iqr.multiply")
				copy = self.parent.config.getParam("outlier.copy.results")
				if copy: 
					cleanColumns = []
					
					numColumns = len(self.parent.grouping.getColumnNames())
					for groupName, columnNames in grouping.items():
						cleanColumns.append(["reO({}):{}".format(groupName,colName) for colName in columnNames.values])

					cleanData = pd.DataFrame(np.zeros(shape=(self.dfs[dataID].index.size,numColumns)), index=self.dfs[dataID].index, columns = np.array(cleanColumns).flatten())
			
				for n, (groupName, columnNames) in enumerate(grouping.items()):
					X = self.dfs[dataID][columnNames].values
					Q = np.nanquantile(X,q=[0.25,0.5,0.75],axis=1)
					Q1 = Q[0,:].reshape(X.shape[0],1)
					M = Q[1,:].reshape(X.shape[0],1)
					Q3 = Q[2,:].reshape(X.shape[0],1)
					IQR = np.abs(Q3-Q1)

					outlierBool = np.logical_or(X > M + m * IQR, X < M - m * IQR)
					
					if not copy:		
						X = self.dfs[dataID][columnNames].values
						X[outlierBool] = np.nan
						self.dfs[dataID].loc[:,columnNames] = X
						
					else:
						X[outlierBool] = np.nan
						cleanData[cleanColumns[n]] = X 

				if not copy:
					return getMessageProps("Done ..","Outlier replaced with NaN.")
				else:
					return self.joinDataFrame(dataID,cleanData)
			
			
			
		else:
			return errorMessage

	def fillNaNBy(self,dataID,columnNames,fillBy = "Row mean"):
		""
		
		X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
		
		#find indices to be replaced
		if fillBy == "Row mean":
			arr = X.values
			nanMean = np.nanmean(arr,axis=1, keepdims = True)
			self.dfs[dataID].loc[X.index,X.columns] = np.where(np.isnan(arr),nanMean,arr)
		elif fillBy == "Row median":
			arr = X.values
			nanMedian = np.nanmedian(arr,axis=1, keepdims = True)
			self.dfs[dataID].loc[X.index,X.columns] = np.where(np.isnan(arr),nanMedian.reshape(-1,1),arr)
		elif fillBy == "Column median":
			self.dfs[dataID].loc[X.index,X.columns] = X.fillna(X.median()) 
		elif fillBy == "Column mean":
			self.dfs[dataID].loc[X.index,X.columns] = X.fillna(X.mean()) 
		else:
			return getMessageProps("Error..","FillBy method not found.")
		return getMessageProps("Done ..","NaN were replaced.")
	

	def imputeNanByModel(self,dataID,columnNames,estimator):
		""
		imputeEstimator = {
							"BayesianRidge":BayesianRidge,
							"DecisionTreeRegressor":DecisionTreeRegressor,
							"ExtraTreesRegressor":ExtraTreesRegressor,
							"KNeighborsRegressor":KNeighborsRegressor
						}
					
		if estimator in imputeEstimator:
			with threadpool_limits(limits=1, user_api='blas'): #require to prevent crash (np.dot not thread safe)
				X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
				imputeEstimator = imputeEstimator[estimator]()

				imputedData = IterativeImputer(estimator=imputeEstimator).fit_transform(X.values)
				df = pd.DataFrame(imputedData,index=X.index,columns = ["Imp({}):{}".format(estimator,colName) for colName in columnNames.values])
				return self.joinDataFrame(dataID,df)
			return errorMessage
	
	def fill_na_in_columnList(self,columnLabelList,id = None, naFill = None):
		'''
		Replaces nan in certain columns by value
		'''
		if naFill is None:
			naFill = self.replaceObjectNan
		if id is None:
			id = self.currentDataFile
		self.dfs[id][columnLabelList] = self.dfs[id][columnLabelList].fillna(naFill)
	
	
	def fill_na_with_data_from_gauss_dist(self,columnLabelList,downshift,width,mode):
		'''
		Replaces nans with random samples from standard distribution. 
		'''
		means = self.df[columnLabelList].mean()
		stdevs =  self.df[columnLabelList].std()
		df = pd.DataFrame()
		for n,numericColumn in enumerate(columnLabelList):
			data = self.df[numericColumn].values
			mu, sigma = means[n], stdevs[n]
			newMu = mu - sigma * downshift
			newSigma = sigma * width
			mask = np.isnan(data)
			data[mask] = np.random.normal(newMu, newSigma, size=mask.sum())
			if mode in ['Replace','Replace & add indicator']:
				self.df[numericColumn] = data
				df['indicImp_{}'.format(numericColumn)] = mask
			else:
				df['imput_{}'.format(numericColumn)] = data
		
		if mode in ['Create new columns','Replace & add indicator']:
			newColumns = self.join_df_to_currently_selected_df(df,exportColumns = True)
			return newColumns
		
	def fit_transform(self,obj,columns,namePrefix = 'Scaled', dropnan=True, transpose=True):
		'''
		Fit and transform data using an object from the scikit library.
		'''
		newColumnNames = [self.evaluate_column_name('{}{}'.format(namePrefix,column), useExact = True)
						 	for column in columns]
		
		df, idx = self.row_scaling(obj,columns,dropnan,transpose)
		df_ = pd.DataFrame(df,index = idx, columns = newColumnNames)
		newColumnNames = self.join_df_to_currently_selected_df(df_, exportColumns = True)
		return newColumnNames
	
			
	def row_scaling(self,obj,columnNames,dropnan, transpose):
		"""
		"""
		if dropnan:
			X = self.df[columnNames].dropna()
		else:
			X = self.df[columnNames]
		idx = X.index
		if transpose:
			X = np.transpose(X.values)
		
		normX = getattr(obj,'fit_transform')(X)
		if transpose:
			return np.transpose(normX), idx
		else:
			return normX, idx
	
	def getCategoricalColumns(self,dataID):
		'''
		Returns columns names that are objects.
		Internal function.
		'''
		if dataID in self.dfs:
			return self.dfsDataTypesAndColumnNames[dataID]["Categories"]
		else:
			return []
			
	def getNonFloatColumns(self,dataID):
		""
		if dataID in self.dfs:
			return pd.concat([self.dfsDataTypesAndColumnNames[dataID]["Categories"] , self.dfsDataTypesAndColumnNames[dataID]["Integers"] ])
		else:
			return []

	def getNumericColumns(self,dataID):
		'''
		Returns columns names that are float and integers
		Function that cannot be called from thread calling
		'''
		if dataID in self.dfs:
			return self.dfsDataTypesAndColumnNames[dataID]['Numeric Floats']

		return []

	def getMinMax(self,dataID,columnName):
		""
		if isinstance(columnName,list):
			columnName = columnName[0]

		if dataID in self.dfs:
			data = self.dfs[dataID][columnName].values
			minValue = np.nanmin(data)
			maxValue = np.nanmax(data)
			return minValue,maxValue

		return np.nan, np.nan

	def getNumValidValues(self,dataID,columnName):
		""
		if isinstance(columnName,list):
			columnName = columnName[0]

		if dataID in self.dfs:
			numValidValue = self.dfs[dataID][columnName].count()
			return numValidValue

		return np.nan, np.nan


	def setDataByIndexNaN(self,dataID,filterIdx,selectedColumns,baseString = "numFil:NaN"):
		""
		if dataID in self.dfs:
			#print(selectedColumns)
			if selectedColumns is None:
				X = self.dfs[dataID][list(filterIdx.keys())]
				for columnName, idx in filterIdx.items():
					X.loc[idx,columnName] = np.nan
				X.columns = ["{}::{}".format(baseString,colName) for colName in X.columns]
			elif isinstance(selectedColumns,dict):
				totalColumns = np.unique(list(selectedColumns.values())).tolist()
				#print(totalColumns)
				X = self.dfs[dataID][totalColumns]
				for columnName, idx in filterIdx.items():
					if columnName in selectedColumns:
						X.loc[idx,selectedColumns[columnName]] = np.nan
			else:
				return errorMessage
			return self.joinDataFrame(dataID,X)
		else:
			return errorMessage

	def setNaNBasedOnCondition(self,dataID,columnNames, belowThreshold = None, aboveThreshold = None):
		""
		print(aboveThreshold,belowThreshold)
		if dataID in self.dfs:
			data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
			filterIdx = {} 
			for columnName in columnNames.values:
				if aboveThreshold is not None:
					filterIdx[columnName] = data[columnName] > aboveThreshold
				elif belowThreshold is not None:
					filterIdx[columnName] = data[columnName] < belowThreshold
			columnNameBaseString = "numFilt(>{}):NaN".format(aboveThreshold) if aboveThreshold is not None else "numFilt(<{}):NaN".format(belowThreshold)
			return self.setDataByIndexNaN(dataID,filterIdx,None,baseString=columnNameBaseString )
		else:
			return errorMessage

	def subsetDataByIndex(self, dataID, filterIdx, subsetName):
		""
		if dataID in self.dfs:
			subsetDf = self.dfs[dataID].loc[filterIdx,:]
			return self.addDataFrame(subsetDf,fileName = subsetName)
		else:
			return errorMessage
			
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

	def setFileNameByID(self,dataID,fileName):
		""
		if dataID in self.fileNameByID:
			self.fileNameByID[dataID] = fileName
			print(self.fileNameByID)	
			completeKwargs = getMessageProps("Renamed.","Data frame renamed.")
			completeKwargs["dfs"] = self.fileNameByID
			return completeKwargs
	
	def getFileNameByID(self,dataID):
		""
		if dataID in self.fileNameByID:
			return self.fileNameByID[dataID]
			
	def get_groups_by_column_list(self,columnList, sort = False):
		'''
		Returns gorupby object of selected columnList
		'''
		
		if isinstance(columnList,list):
			groupByObject = self.df.groupby(columnList,sort = sort)
			
			return groupByObject
		else:
			return
		
	
	def get_next_available_id(self):
		'''
		To provide consistent labeling, use this function to get the id the new df should be added
		'''
		self.dataFrameId += 1
		idForNextDataFrame = 'DataFrame: {}'.format(self.dataFrameId)
		
		return idForNextDataFrame
	
	def getRowNumber(self,dataID):
		'''
		Returns the number of rows.
		'''
		if dataID in self.dfs:
			return self.dfs[dataID].index.size
		else:
			return 	np.nan
	
	def getIndex(self,dataID):
		'''
		Returns the number of rows.
		'''
		if dataID in self.dfs:
			return self.dfs[dataID].index
		else:
			return 	np.nan
	
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

	def get_number_of_unique_values(self,categoricalColumns, id=None):
		"""
		"""
		if id is None:
			id = self.currentDataFile
			
		resultDict = OrderedDict()
		for categoricalColumn in categoricalColumns:
			resultDict[categoricalColumn] = self.dfs[id][categoricalColumn].unique().size
		
		return resultDict

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
				
		
	def join_missing_columns_to_other_df(self, otherDf, id, definedColumnsList = []):
		'''
		'''

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
		
		storedData.index = storedData.index.astype(otherDf.index.dtype)
		if len(columnsMissing) != 0: 
			resultDataFrame = otherDf.join(storedData[columnsMissing])
			return resultDataFrame
		else:
			return otherDf
		
	def matchModSequenceToSites(self,dataID, proteinGroupColumn,modifiedPeptideColumn, fastaFilePath):
		""
		data = self.getDataByColumnNames(dataID=dataID,columnNames=[proteinGroupColumn,modifiedPeptideColumn])["fnKwargs"]["data"]
		modPeptideSequence = data[modifiedPeptideColumn].values
		proteinGroups = data[proteinGroupColumn].values
		modFinder = ICModPeptidePositionFinder(None,None)
		modFinder.loadFasta(fastaFilePath)
		matchedSiteData = modFinder.matchModPeptides(proteinGroups,modPeptideSequence,data.index.values.tolist())
		return self.joinDataFrame(dataID,matchedSiteData)
		
		#eturn getMessageProps("Done..","Modified peptides matched to sites.")
		
	def meltData(self,dataID,columnNames):
		'''
		Melts data frame.
		'''
		
		if dataID in self.dfs:
			
			idVars = [column for column in self.dfs[dataID].columns if column not in columnNames.values] #all columns but the selected ones
			valueName = self.evaluateColumnName(['melt_value{}'.format(mergeListToString(columnNames)).replace("'",'')], dataID = dataID)[0]
			variableName = self.evaluateColumnName(['melt_variable{}'.format(mergeListToString(columnNames)).replace("'",'')], dataID = dataID)[0]		
			meltedDataFrame = pd.melt(self.dfs[dataID], id_vars = idVars, value_vars = columnNames,
									var_name = variableName,
									value_name = valueName)
			## determine file name
			baseFile = self.getFileNameByID(dataID)
			numMeltedfiles = len([fileName for fileName in self.fileNameByID.values() if 'Melted_' in fileName])			
			fileName =  'melt({})_{}'.format(baseFile,numMeltedfiles)	

			return self.addDataFrame(meltedDataFrame, fileName=fileName)					
		else:
			return errorMessage


	def correlateDfs(self,corrParams):
		""
		dataID1 = corrParams["dataID1"]
		dataID2 = corrParams["dataID2"]
		columnNames1 = corrParams["columnNames1"] if not corrParams["columnNames1"].empty else self.getNumericColumns(dataID1)
		columnNames2 = corrParams["columnNames2"] if not corrParams["columnNames2"].empty else self.getNumericColumns(dataID2)
		ignoreIndex = corrParams["ignoreIndex"]
		if dataID1 in self.dfs and dataID2 in self.dfs:

			df = self.getDataByColumnNames(dataID=dataID1,columnNames=columnNames1)["fnKwargs"]["data"]
			otherDf = self.getDataByColumnNames(dataID=dataID2,columnNames=columnNames2)["fnKwargs"]["data"]
			if ignoreIndex and corrParams["axis"] == 1:
				df.columns = np.arange(df.columns.size)
				otherDf.columns = np.arange(otherDf.columns.size)
			print(df, otherDf)
			print(df.corrwith(otherDf, axis=corrParams["axis"],method=corrParams["method"]))
			return {}
		else:
			return errorMessage

	def _getCorrParamFromDict(self,corrParams):
		dataID1 = corrParams["dataID1"]
		dataID2 = corrParams["dataID2"]
		columnNames1 = corrParams["columnNames1"] if not corrParams["columnNames1"].empty else self.getNumericColumns(dataID1)
		columnNames2 = corrParams["columnNames2"] if not corrParams["columnNames2"].empty else self.getNumericColumns(dataID2)

		return dataID1,dataID2,columnNames1,columnNames2
	
	def correlateEachFeatureOfTwoDfs(self,corrParams):
		""
		dataID1,dataID2,columnNames1,columnNames2 = self._getCorrParamFromDict(corrParams)
		if any(x.size < 3 for x in [columnNames2,columnNames1]):
			return getMessageProps("Error..","Each data frame must have more than three columns.")
		else:
			df1 = self.getDataByColumnNames(dataID=dataID1,columnNames=columnNames1)["fnKwargs"]["data"].dropna(thresh=3)
			df2 = self.getDataByColumnNames(dataID=dataID2,columnNames=columnNames2)["fnKwargs"]["data"].dropna(thresh=3)

			catCol1 = self.getCategoricalColumns(dataID1)
			catCol2 = self.getCategoricalColumns(dataID2)

			catData1 = self.getDataByColumnNames(dataID=dataID1,columnNames=catCol1)["fnKwargs"]["data"].loc[df1.index,:]
			catData2 = self.getDataByColumnNames(dataID=dataID2,columnNames=catCol2)["fnKwargs"]["data"].loc[df2.index,:]

			A = pearsonByRowsTwoArray(df1.values,df2.values)
			
			
			catRep2 = np.repeat(catData1.values,df2.index.size,axis=0)
			catRep1 = np.tile(catData2.values,(df1.index.size,1))
			
			columnNames = ["r","p"] + ["{}_x".format(colName) for colName in catData2.columns.tolist()] + ["{}_y".format(colName) for colName in catData1.columns.tolist()]
			resultDf = pd.concat([
				pd.DataFrame(A,columns=["r","p"]),
				pd.DataFrame(catRep1, columns = catData2.columns),
				pd.DataFrame(catRep2, columns= catData1.columns)], axis=1, ignore_index=True)
			resultDf.columns = columnNames 
			return self.addDataFrame(resultDf,fileName="rowByRowCorr:{}:{}".format(self.getFileNameByID(dataID1),self.getFileNameByID(dataID2)))

	def correlateFeaturesDfs(self,corrParams):
		""
		dataID1,dataID2,columnNames1,columnNames2 = self._getCorrParamFromDict(corrParams)

		if dataID1 in self.dfs and dataID2 in self.dfs:
			df = self.getDataByColumnNames(dataID=dataID1,columnNames=columnNames1)["fnKwargs"]["data"]
			otherDf = self.getDataByColumnNames(dataID=dataID2,columnNames=columnNames2)["fnKwargs"]["data"]
			
			if df.shape[0] == otherDf.shape[0]:
				result = self.statCenter.correleateColumnsOfTwoDfs(df,otherDf)
				result = result.reset_index()
				return self.addDataFrame(result,fileName="Correlated")
		else:
			return errorMessage

	def mergeDfs(self,mergeParams, how = "left", indicator = True):
		""

		leftDataID = mergeParams["left"]["dataID"]
		rightDataID = mergeParams["right"]["dataID"]

		leftMergeColumn = mergeParams["left"]["mergeColumns"].values.tolist()
		rightMergeColumn = mergeParams["right"]["mergeColumns"].values.tolist()
		
		if len(leftMergeColumn) != len(rightMergeColumn):

			return getMessageProps("Error ..","Merge column selection of different size.")
		
		

		leftSelectedColumnNames = mergeParams["left"]["columnNames"] if mergeParams["left"]["selectedColumns"].empty else mergeParams["left"]["selectedColumns"]
		rightSelectedColumnNames = mergeParams["right"]["columnNames"] if mergeParams["right"]["selectedColumns"].empty else mergeParams["right"]["selectedColumns"]

		leftKeepColumnNames = leftMergeColumn + [colName for colName in leftSelectedColumnNames.values if colName not in leftMergeColumn]
		rightKeepColumnNames = rightMergeColumn + [colName for colName in rightSelectedColumnNames.values if colName not in rightMergeColumn]

		if leftDataID in self.dfs and rightDataID in self.dfs:

			leftDf = self.dfs[leftDataID][leftKeepColumnNames] #subset data by selection 
			rightDf = self.dfs[rightDataID][rightKeepColumnNames] #subset data by selection 

		
			mergedDataFrames = leftDf.merge(rightDf,
								how = how, 
								left_on = leftMergeColumn, 
								right_on = rightMergeColumn, 
								indicator = indicator)

			if "_merge" in mergedDataFrames.columns:
				mergedDataFrames["_merge"] = mergedDataFrames["_merge"].astype(str)
			mergedDataFrames.reset_index(drop=True,inplace=True)
			return self.addDataFrame(
							dataFrame = mergedDataFrames, 
							fileName = "merged({}:{})".format(self.fileNameByID[leftDataID],self.fileNameByID[rightDataID])
							)

		return errorMessage


	def melt_data_by_groups(self,columnGroups, id = None):
		'''
		Melts multiple subsets of data to one df
		'''
		if id is not None:
			self.set_current_data_by_id(id)
			
		meltedSubsets = []
		columnNames = []
		
		groupColumns = []
		# get columns that will be used for merging
		[groupColumns.extend(columns) for columns in columnGroups.values()]
		for n,(groupId, columnList) in enumerate(columnGroups.items()):
			#almost impossible to occur from instant clue but rather check this
			if any(column not in self.df.columns for column in columnList):
				continue
			if n == 0:
				idVars = [column for column in self.df.columns if column not in groupColumns]
			else:
				idVars = None			
			
			meltDf = pd.melt(self.df, value_vars = columnList,
							id_vars = idVars, 
							value_name = 'Values_{}'.format(groupId),
							var_name = 'Variable_{}'.format(groupId))
			columnNames.extend(meltDf.columns.values.tolist())
			meltedSubsets.append(meltDf)
		
		if len(meltedSubsets) > 1:
			conDf = pd.concat(meltedSubsets,axis=1,ignore_index=True)
			conDf.columns = columnNames
			return conDf
		else:
			return None


	def pivotTable(self,dataID, indexColumn, columnNames, findCommonString = True):
		""
		if dataID in self.dfs:
			if findCommonString:
				commonString = findCommonStart(*self.getUniqueValues(dataID,columnNames))
			else:
				commonString = ""
			data = pd.pivot_table(data = self.dfs[dataID], columns=[columnNames], index=[indexColumn])
			mergedColumns = ['_'.join(col).strip().replace(commonString,"") for col in data.columns.values]
			data.columns = mergedColumns
			data = data.reset_index()
			fileName = "PivotT({}):({})".format(columnNames,self.getFileNameByID(dataID))			
			return self.addDataFrame(data, fileName=fileName)
		else:
			return errorMessage

		
	def unstackColumn(self,dataID, columnNames, separator = ';'):
		'''
		Unstacks column. 
		Depracted and replaced by explode since 0.11.
		'''
		return {}
		# if dataID in self.dfs:
		# 	columnName = columnNames.values[0]
		# 	row_accumulator = []
			
		# 	def splitListToRows(row, separator):
		# 		split_row = row[columnName].split(separator)
				
		# 		for s in split_row:
		# 			new_row = row.to_dict()
		# 			new_row[columnName] = s
		# 			row_accumulator.append(new_row)    

			
		# 	self.dfs[dataID].apply(splitListToRows, axis=1, args = (separator, ))
		# 	unstackedDf = pd.DataFrame(row_accumulator)
			
		# 	#acquire name and source file
			
		# 	baseFile = self.getFileNameByID(dataID)
		# 	fileName = 'unstack({})[{}]'.format(columnName,baseFile)
			
		# 	completeKwargs = self.addDataFrame(unstackedDf, fileName = fileName)				
		# 	return completeKwargs
		
	def transposeDataFrame(self,dataID, columnNames = None, columnLabel = None):
		""
		if dataID in self.dfs:

			newColumnNames = self.dfs[dataID][columnLabel].values.flatten()
			if columnNames is not None and np.unique(newColumnNames).size != columnNames.size:
				newColumnNames = ["{}_{}".format(newColumnNames[n],n) for n in np.arange(newColumnNames.size)]

			if columnLabel is not None:
				if columnNames is not None:
					requiredColumnNames = [colName for colName in self.dfs[dataID].columns if colName != columnLabel and colName in columnNames.values]
				else:
					requiredColumnNames = [colName for colName in self.dfs[dataID].columns if colName != columnLabel]
			else:
				requiredColumnNames = self.dfs[dataID].columns.values 
	
			dataT = self.dfs[dataID][requiredColumnNames].T 
			dataT.columns = newColumnNames
			dataT = dataT.reset_index()

			return self.addDataFrame(dataT,fileName="t:{}".format(self.getFileNameByID(dataID)))

		return errorMessage

	def transform_data(self,columnNameList,transformation):
		'''
		Calculates data transformation and adds these to the data frame.
		'''	
		newColumnNames = [self.evaluate_column_name('{}_{}'.format(transformation,columnName)) \
		for columnName in columnNameList]
		
		if transformation == 'Z-Score_row':
			transformation = 'Z-Score'
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
	

	def updateData(self,dataID,data):
		'''
		Updates dataframe, input: dataID and data
		'''
		if dataID in self.dfs:
			self.dfs[dataID] = data
			self.extractDataTypeOfColumns(dataID)
			
			return getMessageProps("Updated..","Data ({}) updated.".format(dataID))
		else:
			return errorMessage

	def update_columns_of_current_data(self):
		'''
		Updates the variable: self.df_columns and renews the data type - column relationship.
		'''	
		#self.df_columns = self.df.columns.values.tolist() 
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
		
		
		
			
	def sortData(self, dataID, columnNames, kind = 'mergesort', ascending = True, na_position = 'last'):
		'''
		Sort rows in one or multiple columns.
		'''
		if dataID in self.dfs:
			if dataID in self.rememberSorting:
				
				columnNameThatWereSortedAscending = self.rememberSorting[dataID]			
				changeToDescending = [col for col in columnNames.values if col in columnNameThatWereSortedAscending]
				## Check if all columns were sorted already in ascending order 
				numToDescending = len(changeToDescending)
				if numToDescending == columnNames.size and numToDescending > 0:
					ascending = False
					columnNameThatWereSortedAscending = [col for col in columnNameThatWereSortedAscending if col not in columnNames.values] 
					self.rememberSorting[dataID] = columnNameThatWereSortedAscending
		
			self.dfs[dataID].sort_values(by = columnNames.values.tolist(), 
										 kind= kind,
										 ascending = ascending,
										 na_position = na_position,
										 inplace = True)
			
			if ascending:
				## save columns that were already sorted 
				if dataID in self.rememberSorting:
				
					columnNameThatWereSortedAscending = self.rememberSorting[dataID]
					columnNamesToAdd = [col for col in columnNames.values if col not in columnNameThatWereSortedAscending] 
					columnNameThatWereSortedAscending.extend(columnNamesToAdd)
					self.rememberSorting[dataID] = columnNameThatWereSortedAscending
						
				else:
					self.rememberSorting[dataID] = columnNames.values.tolist()
							
			return getMessageProps("Sorting..",
				"Columns were sorted in {} order.\nIf you sort again, the order will be reversed.".format("ascending" if ascending else "descending"))	
		else:
			return errorMessage	
		
	def sortDataByValues(self,dataID,columnName,values):
		""
		if dataID in self.dfs:
			if not isinstance(columnName,str):
				return getMessageProps("Error..","Attribute column name must be string.")

			indexSorter = dict(zip(values,range(values.size)))
			indexMap = self.dfs[dataID][columnName].map(indexSorter)
			sortedIndex = indexMap.sort_values().index
			return self.sortRowsByIndex(dataID,sortedIndex)

		else:
			return errorMessage

	def sortRowsByIndex(self,dataID,sortedIndex):
		"Does not perform checking of correct index. Need to be done before."
		if dataID in self.dfs:
			self.dfs[dataID] = self.dfs[dataID].loc[sortedIndex,:]
			return getMessageProps("Sorted..","Rows sorted.")
		else:
			return errorMessage

	def splitColumnsByString(self,dataID,columnNames,splitString):
		""
		if dataID in self.dfs:
			for columnName in columnNames.values:

				splitColumns = self.dfs[dataID][columnName].str.split(splitString, expand = True)
				nColumns = splitColumns.columns.size
				if nColumns == 1:
					return getMessageProps("Error ..","Column {} does not contain split string. Aborting.".format(columnName))
				newColumnNames = ["S({}):{}_{:02d}".format(splitString,columnName,n) for n in range(nColumns)]
				splitColumns.columns = newColumnNames
				splitColumns.fillna(self.getNaNString(),inplace=True)
				self.joinDataFrame(dataID,splitColumns)

			completeKwargs = getMessageProps("Done..","Column(s) was/were split on split string: {}".format(splitString))
			completeKwargs["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			return completeKwargs
		else:
			return errorMessage

			
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
	
	def aggregateNRows(self,dataID,columnNames, n, metric='mean'):
		
		if dataID in self.dfs:
			
			groupedData = self.dfs[dataID][columnNames].groupby(self.dfs[dataID].index // n * n)
			if hasattr(groupedData,metric):
				dataFrame = getattr(groupedData,metric)()
				fileName = "Agg({}:{}):{}".format(n,metric,self.getFileNameByID(dataID))
				return self.addDataFrame(dataFrame,fileName=fileName)
			else:
				return getMessageProps("Error ..","Metric unknown.")
		else:
			return errorMessage
		
			
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

		
		
			
		
				
		
	
		
		
		
	
	
	
	
	
	
		
		










