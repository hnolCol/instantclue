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

import time
import re 
import os 
from multiprocessing import Pool
from collections import OrderedDict
from itertools import compress, chain, groupby
import numpy as np
import pandas as pd

from matplotlib.colors import to_hex

from sklearn.experimental import enable_iterative_imputer
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

from scipy.signal import lfilter
from scipy.stats import pearsonr, gaussian_kde

from ..utils.stringOperations import getMessageProps, getReadableNumber, mergeListToString, findCommonStart, getRandomString
from ..filter.categoricalFilter import CategoricalFilter
from ..filter.numericalFilter import NumericFilter
from ..color.colorManager import ColorManager
from ..statistics.statistics import StatisticCenter
from ..normalization.normalizer import Normalizer
from ..transformations.transformer import Transformer
from ..proteomics.ICModifications import ICModPeptidePositionFinder
from .ICExcelExport import ICDataExcelExporter, ICHClustExporter
from .ICSmartReplace import ICSmartReplace

import numba as nb
from numba import prange, jit
from numba.core.decorators import njit
from numba.np.ufunc import parallel

from typing import List, Tuple, Iterable, Dict


FORBIDDEN_COLUMN_NAMES = ["color","size","idx","layer","None"]

def fasta_iter(fasta_name):
    """
    modified from Brent Pedersen
    Correct Way To Parse A Fasta File In Python
    given a fasta file. yield tuples of header, sequence
    """
    
    fh = open(fasta_name)
    faiter = (x[1] for x in groupby(fh, lambda line: line[0] == ">"))

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


def rowByRow(X,Y,chunkIdx = None):
	""
	nRows = X.shape[0] * Y.shape[0]
	idx = 0
	A = np.empty(shape=(nRows,2), dtype=np.float64)
	for n in range(X.shape[0]):
		for m in range(Y.shape[0]):
			A[idx] =  pearsonWithNaNCheck(X[n],Y[m])
			idx += 1
	return (chunkIdx,A)


def pearsonWithNaNCheck(x,y):
	"Checks nan in both arrays."
	nonNaNIdx = [idx for idx in range(y.size) if not np.isnan(x[idx]) and not np.isnan(y[idx])]
	if len(nonNaNIdx) > 2:
		return pearsonr(x[nonNaNIdx],y[nonNaNIdx])
	return [np.nan,np.nan]

def pearsonByRowsTwoArray(X,Y,NProcesses = 8):
	"Correlate rows of two arrays using multiprocessing. The bigger array will be split into chunks."
	#chunk bigger array 
	if X.shape[0] > Y.shape[0]:
		chunks = np.array_split(X,NProcesses,axis=0)
	else:
		chunks = np.array_split(Y,NProcesses,axis=0)
	
	with Pool(NProcesses) as p:
		rs = p.starmap(rowByRow,[(chunk,Y,chunkIdx) for chunkIdx, chunk in enumerate(chunks)])
		rs.sort(key=lambda x: x[0]) 
		A = np.concatenate([r[1] for r in rs],axis=0)
	return A


def corr2_coeff(A, B):
	# Rowwise mean of input arrays & subtract from input arrays themeselves
	nRows = A.shape[0] * B.shape[0]
	X = np.zeros(shape=(nRows,1))

	A_mA = A - np.nanmean(A, axis=1)[:, None]
	B_mB = B - np.nanmean(B,axis=1)[:, None]
	
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
		self.tooltipData = OrderedDict()
		self.dfShapes = OrderedDict()
		self.dfsDataTypesAndColumnNames = OrderedDict() 
		self.fileNameByID = OrderedDict() 
		self.excelFileIO = OrderedDict()
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
	

	def setPlotter(self,plotter) -> None:
		""
		self.Plotter = plotter

	def addAnnotationColumnByIndex(self,dataID : str, indices : pd.Index, columnName : str) -> dict:
		""
		if dataID in self.dfs:
			columnName = self.evaluateColumnName(columnName,dataID=dataID)
			annotationData = pd.Series(
					[self.getNaNString()] * self.dfs[dataID].index.size, 
					index = self.dfs[dataID].index,
					name = columnName)
			annotationData.loc[indices] = "+"
			df = self.dfs[dataID].join(annotationData)
			self.updateData(dataID,df)

			funcProps = getMessageProps("Column added","Column {} was added to data.".format(columnName))
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			funcProps["tooltipData"] = self.getTooltipdata(dataID)

			return funcProps
		else:
			return errorMessage


	def addColumnData(self,dataID,columnName,columnData,rowIndex = None, evaluateName = True,**kwargs):
		'''
		Adds a new column to the data
		'''
		if evaluateName:
			columnName = self.evaluateColumnName(columnName,dataID=dataID,**kwargs)
		if rowIndex is None:
			self.dfs[dataID].loc[:,columnName] = columnData
		else:
			self.dfs[dataID].loc[rowIndex,columnName] = columnData
		self.extractDataTypeOfColumns(dataID)
		funcProps = getMessageProps("Column added","Column {} was added to data.".format(columnName))
		funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
		funcProps["tooltipData"] = self.getTooltipdata(dataID)
		return funcProps

	def addIndexColumn(self,dataID : str) -> dict:
		""
		if dataID in self.dfs:
			dfShape, rowIdx = self.getDataFrameShape(dataID)
			numRows, _ = dfShape
			columnName = "Index"
			idxData = pd.DataFrame(np.arange(numRows), index=rowIdx, columns=[columnName])
			return self.joinDataFrame(dataID,idxData)
		else:
			return errorMessage
	
	def addGroupIndexColumn(self,dataID : str, columnNames : pd.Series):
		""
		if dataID in self.dfs:
			#dfShape, rowIdx = self.getDataFrameShape(dataID)
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
			print(e)
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
			
	def areAllColumnsInData(self,dataID : str,columnNames : pd.Series) -> bool:
		""
		if dataID in self.dfs:
			return columnNames.isin(self.dfs[dataID].columns.array).all()
		return False

	def readExcelFile(self,pathToFiles : str) -> dict:
		""
		if hasattr(self,"excelFileIO"):
			self.excelFileIO = OrderedDict()
		fileSheetNames = OrderedDict([("Sheet",[]),("File",[])])
		for filePath in pathToFiles:
			fileName = os.path.basename(filePath)
			if fileName in self.excelFileIO and filePath == self.excelFileIO[fileName]["path"]:
				sheetNames = self.excelFileIO[fileName]["excelFile"].sheet_names
				fileSheetNames["File"].extend([fileName]*len(sheetNames))
				fileSheetNames["Sheet"].extend(sheetNames)
				continue
			elif fileName in self.excelFileIO and filePath != self.excelFileIO[fileName]["path"]:
				fileName = fileName + getRandomString(N=6)
			self.excelFileIO[fileName] = {}
			excelFile = pd.ExcelFile(filePath)
			sheetNames = np.array(excelFile.sheet_names)
			self.excelFileIO[fileName]["sheetNames"] = sheetNames
			self.excelFileIO[fileName]["path"] = filePath
			self.excelFileIO[fileName]["excelFile"] = excelFile

			fileSheetNames["File"].extend([fileName]*len(sheetNames))
			fileSheetNames["Sheet"].extend(sheetNames)
		
		return {"fileSheetNames":{"df":pd.DataFrame().from_dict(fileSheetNames)}}


	def readExcelSheetFromFile(self,ioAndSheets,props,instantClueImport):
		""
		#df = pd.read_excel(self.excelFiles[fileName]['excelFile'],sheetName)
		groupings = OrderedDict()
		for dataFrameName, readExcelProps in ioAndSheets.items():
			if readExcelProps["io"] in self.excelFileIO:
				excelFile = self.excelFileIO[readExcelProps["io"]]["excelFile"]
				if instantClueImport:
					params = pd.read_excel(excelFile,sheet_name="Software Info", index_col="Parameters")
					numberOfGroupings = int(float(params.loc["Groupings"]))
					df = pd.read_excel(excelFile,sheet_name=readExcelProps["sheet_name"],skiprows=numberOfGroupings).dropna(axis=1,how="all")
					funcProps = self.addDataFrame(df,fileName=dataFrameName, cleanObjectColumns=True)
					#groupings
					
					try:
						if numberOfGroupings > 0:
							groupingValues = pd.read_excel(excelFile,sheet_name=readExcelProps["sheet_name"],nrows=numberOfGroupings+1,header=None,index_col=0).dropna(axis=1,how="all")
							
							for n,groupingName in enumerate(groupingValues.index):
								if n == groupingValues.index.size-1:
									continue
								
								X = pd.DataFrame(groupingValues.values[[n,-1],:].T,columns=["groupName","columnName"]).dropna()
								if X.index.size > 0:
									grouping = OrderedDict([(groupName,groupData["columnName"]) for groupName, groupData in X.groupby(by="groupName",sort=False)])
									if groupingName not in groupings:
										groupings[groupingName] = grouping
									else:
										replaceGrouping = OrderedDict()
										for groupName, groupedItems in groupings[groupingName].items():
											if groupName in grouping:
												groupedItems = np.concatenate([groupedItems.values,grouping[groupingName].values])
												
											replaceGrouping[groupName] = groupedItems
										groupings[groupingName] = groupedItems
							
					except:
						funcProps["messageProps"] = getMessageProps("Error","File was loaded, and grouping detected. However the grouping could not be loaded.")

					funcProps["groupings"] = groupings
				else:
					props = self.checkLoadProps(props)
					df = pd.read_excel(excelFile,sheet_name=readExcelProps["sheet_name"],**props)
					funcProps = self.addDataFrame(df,fileName=dataFrameName,cleanObjectColumns=True)

		return funcProps


	def addDataFrame(self,dataFrame : pd.DataFrame, dataID = None, fileName : str = '', 
							cleanObjectColumns : bool = False) -> dict:
		'''
		Adds new dataFrame to collection.
		'''
		if dataID is None:
			dataID  = self.getNextDataFrameID()
		dataFrame = self.checkForInternallyUsedColumnNames(dataFrame)
		self.dfs[dataID] = dataFrame
		self.tooltipData[dataID] = dict()
		self.extractDataTypeOfColumns(dataID)
		self.saveFileName(dataID,fileName)
		
		rows,columns = self.dfs[dataID].shape

		#clean up nan in object columns
		if cleanObjectColumns:
			objectColumnList = self.dfsDataTypesAndColumnNames[dataID]["Categories"]
			self.fillNaInObjectColumns(dataID,objectColumnList)

		return {"messageProps":
				{
					"title":"Data Frame Loaded {}".format(fileName),
					"message":"{} loaded and added.\nShape (rows x columns) is {} x {}".format(dataID,rows,columns)
				},
				"columnNamesByType":self.dfsDataTypesAndColumnNames[dataID],
				"dfs":self.fileNameByID
			}

	def addDataFrames(self,fileNameAndDataFrame : List[tuple], copyTypesFromDataID : str) -> int:
		"Not used from outside (e.g. new data frame)"
		if copyTypesFromDataID in self.dfsDataTypesAndColumnNames:
			for fileName,dataFrame in fileNameAndDataFrame:
				dataID  = self.getNextDataFrameID()
				self.dfs[dataID] = dataFrame
				self.saveFileName(dataID,fileName)
				#takes longer, but otherwise there is no tooltip.
				self.extractDataTypeOfColumns(dataID)
				# this is faster; 
				#self.dfsDataTypesAndColumnNames[dataID] = self.dfsDataTypesAndColumnNames[copyTypesFromDataID].copy()
				
			return len(fileNameAndDataFrame)
		return 0 
	
	def checkForInternallyUsedColumnNames(self,dataFrame : pd.DataFrame) -> pd.DataFrame:
		""
		
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

	def getQuickSelectData(self, dataID : str,filterProps : dict) -> dict:
		""
		if dataID in self.dfs:
			if all(filterProp in filterProps for filterProp in ["columnName","mode","sep"]):
				columnName = filterProps["columnName"]
				
				if filterProps["mode"] == "unique":
					
					sep = filterProps["sep"]
					#getUniqueCategroies returns a data frame, therefore index column 
					#to get a pandas Series (QuickSelect Model works with series)
					data = self.categoricalFilter.getUniqueCategories(dataID,columnName,splitString=sep)
					#returns a dict with error message if not successfull
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

	def getDataFrameShape(self, dataID : str) -> Tuple[int,int]:
		"Returns the shape of the data frame, taking clipping into account."
		if dataID in self.dfs:
			if dataID in self.clippings:
				rowIdx = self.clippings[dataID]
				return self.dfs[dataID].loc[rowIdx,:].shape, rowIdx
			else:
				return self.dfs[dataID].shape, self.dfs[dataID].index
		else:
			return (0,0)

	def groupbyAndAggregate(self,dataID : str, columnNames : pd.Series, groupbyColumn : str, metric : str ="mean"):
		"""
		Aggregates data by a specific groupbyColumn(s). 
		ColumnNames can only be numeric.
		"""
		if dataID in self.dfs:
			requiredColumns =  pd.concat([columnNames,pd.Series(groupbyColumn.values.flatten())],ignore_index=True) 
			data = self.getDataByColumnNames(dataID,requiredColumns)["fnKwargs"]["data"]
			if metric == "text-merge":
				aggregatedData = data.groupby(by=groupbyColumn.values.flatten().tolist(),sort=False)[columnNames].agg(lambda x: ";".join(list(x))).reset_index()
			else:
				aggregatedData = data.groupby(by=groupbyColumn.values.flatten().tolist(),sort=False).aggregate(metric).reset_index()
			#colnames that are not string, must be tuples - merge them to string.
			columnNames = [colName if isinstance(colName,str) else "_".join(colName) for colName in aggregatedData.columns.to_list()]
			aggregatedData.columns = columnNames
			return self.addDataFrame(aggregatedData,fileName = "{}(groupAggregate({}:{})".format(metric,self.getFileNameByID(dataID),mergeListToString(groupbyColumn.values)))
		else:
			return errorMessage

	def checkColumnNamesInDataByID(self,dataID : str,columnNames : pd.Series) -> List[str]:
		""
		checkedColumnNames = []

		if isinstance(columnNames,str):
			columnNames = [columnNames]

		if not isinstance(columnNames,list):
			raise ValueError("Provide either list or string")
		
		if dataID in self.dfs:
			checkedColumnNames = [columnName for columnName in columnNames if columnName in self.dfs[dataID].columns] 
		
		return checkedColumnNames

	def loadDefaultReadFileProps(self) -> dict:
		""
		config = self.parent.config 
		props = {
			"encoding":config.getParam("load.file.encoding"),
			"sep":config.getParam("load.file.column.separator"),
			"thousands":config.getParam("load.file.float.thousands"),
			"decimal": config.getParam("load.file.float.decimal"),
			"skiprows": config.getParam("load.file.skiprows"),
			"na_values":config.getParam("load.file.na.values"),
			"index_col":False}
		return self.checkLoadProps(props)

	def checkLoadProps(self,loadFileProps) -> dict:
		""
		if loadFileProps is None:
			return {"sep":"\t"}
		if "sep" in loadFileProps and loadFileProps["sep"] in ["tab","space"]:
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

	
	def columnRegExMatches(self,dataID : str,columnNames : pd.Series, searchString : str, splitString : str =";"):
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

	def copyDataFrameSelection(self,dataID : str, columnNames : pd.Series):
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
		data.to_clipboard(sep=self.parent.config.getParam("export.file.clipboard.separator"))
		return getMessageProps("Done ..","Quick Select data copied to clipboard. Data might contain duplicated rows.")

	def copyDataFrameToClipboard(self,dataID=None,data=None,attachDataToMain = None):
		""
		sepForExport=self.parent.config.getParam("export.file.clipboard.separator")
		if dataID is None and data is None:
			return {"messageProps":{"title":"Error",
								"message":"Neither id nor data specified.."}
					}	
		elif dataID is not None and dataID in self.dfs and attachDataToMain is not None:
			#attach new data to exisisitng
			dataToCopy = attachDataToMain.join(self.dfs[dataID])
			dataToCopy.to_clipboard(sep=sepForExport)

		elif dataID is not None and dataID in self.dfs:
			data = self.getDataByDataID(dataID)
			data.to_clipboard(sep=sepForExport)

		elif isinstance(data,pd.DataFrame):

			data.to_clipboard(sep=sepForExport)
		
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
			return getMessageProps("Error ..","There was an error loading the file from clipboard." + str(e))

		localTime = time.localtime()
		current_time = time.strftime("%H_%M_%S", localTime)

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


	def randomSelection(self, dataID, N,*args,**kwargs):
		""
		sampleIdx = self.getDataByDataID(dataID).sample(n=N).index
		return self.addAnnotationColumnByIndex(dataID,sampleIdx,"RandomSelection(N={})".format(N))

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
			taskCompleteKwargs = getMessageProps("Column(s) deleted.","Data tree updated.")
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
			if "_" in evalColumnName and evalColumnName.split("_")[-1].isdigit():
				removeChar = len(evalColumnName.split("_")[-1])
				count += 1
				evalColumnName = evalColumnName[:-removeChar] + "{:02d}".format(count)
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
				if dataID not in self.tooltipData:
							self.tooltipData[dataID] = OrderedDict()
				if dataType != 'object':
					dfWithSpecificDataType = self.dfs[dataID].select_dtypes(include=[dataType])
					
					quantiles = dfWithSpecificDataType.quantile(q=[0,0.25,0.5,0.75,1])
					nanSum = dfWithSpecificDataType.isna().sum()
					fracs = nanSum / dfWithSpecificDataType.index.size
					for columnHeader in dfWithSpecificDataType.columns:
						nans = nanSum[columnHeader]
						frac = round(fracs[columnHeader]*100,1)
						self.tooltipData[dataID][columnHeader] = "Min : {}\n25% Quantile : {}\nMedian : {}\n75% Quantile : {}\nMax : {}\n#-nans : {} ({}%)".format(*[getReadableNumber(x) for x in quantiles[columnHeader].values.flatten()],nans,frac)
				else:
					dfWithSpecificDataType = self.dfs[dataID].select_dtypes(exclude=['float64','int64'])
					
					for columnHeader in dfWithSpecificDataType.columns:
						self.tooltipData[dataID][columnHeader] = "#unique values = {}".format(dfWithSpecificDataType[columnHeader].unique().size)
			except ValueError:
				dfWithSpecificDataType = pd.DataFrame() 		
			columnHeaders = dfWithSpecificDataType.columns.to_list()
			dataTypeColumnRelationship[dTypeConv[dataType]] = pd.Series(columnHeaders, dtype=str)
				
		self.dfsDataTypesAndColumnNames[dataID] = dataTypeColumnRelationship	
	
	def exportDataToExcel(self,pathToExcel,fileNames,dataFrames,softwareParams,groupings=None):
		""
		groupingDetails = dict()
		if groupings is not None and isinstance(groupings,list) and len(groupings) > 0:
			groupingDetails["groupings"] = self.parent.grouping.getGroupingsByList(groupings)
			groupingDetails["colors"] = self.parent.grouping.getGroupColorsByGroupingList(groupings)
		
		exporter = ICDataExcelExporter(pathToExcel,dataFrames,fileNames,softwareParams,groupingDetails)
		exporter.export()
		return getMessageProps("Saved ..","Excel file saved.")

	def exportHClustToExcel(self,
							dataID,
							pathToExcel,
							clusteredData,
							colorArray,
							totalRows,
							clusterLabels,
							clusterColors,
							quickSelectData,
							hclustParams,
							groupings=None,
							colorData = None,
							colorDataArray = None,
							colorColumnNames = []):
		""

		dataColumns = self.getPlainColumnNames(dataID).values.tolist()
		clusterColumns = clusteredData.columns.to_list() 
		extraDataColumns = [columnName for columnName in dataColumns if columnName not in clusterColumns]
		
		columnHeaders = ["Cluster ID"] + clusterColumns + colorColumnNames + extraDataColumns
		groupingDetails = dict()
		rowIdx = clusteredData.index
		extraData = self.getDataByColumnNames(dataID,extraDataColumns,rowIdx=rowIdx)["fnKwargs"]["data"]
		if quickSelectData is not None:
			extraData["QuickSelect"] = np.full(extraData.index.size,"")
			extraData.loc[quickSelectData[0]["dataIndexInClust"],"QuickSelect"] = [to_hex(c) for c in quickSelectData[1]]
			columnHeaders.append("QuickSelect")
		if groupings is not None and isinstance(groupings,list) and len(groupings) > 0:
			
			groupingDetails["groupings"] = self.parent.grouping.getGroupingsByList(groupings)
			groupingDetails["colors"] = self.parent.grouping.getGroupColorsByGroupingList(groupings)
	
		exporter = ICHClustExporter(pathToExcel,clusteredData,columnHeaders,colorArray,totalRows,extraData,clusterLabels,clusterColors,hclustParams,groupingDetails,colorData,colorDataArray,colorColumnNames)
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
			columnNames = self.getPlainColumnNames(dataID) #get all column names
			data = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"] #required to account for grouping.
			if len(columnList) == 1:
				#unpack list, future warning pandas
				columnList = columnList[0]
			groupByObject = data.groupby(columnList,sort = sort,as_index=as_index)
			return groupByObject

	def getTooltipdata(self,dataID : str) -> dict:
		""
		return self.tooltipData[dataID] if dataID in self.tooltipData else {}

	def getColumnNamesByDataID(self,dataID : str) -> dict:
		"Returns Dict of Column Names per DataFrame and Column Type (float,int,string)"
		
		if dataID in self.dfsDataTypesAndColumnNames:
			return {"messageProps":
						{"title":"Updated",
						"message":"Data Frame Selection updated."},
					"dataID" : dataID,
					"columnNamesByType":self.dfsDataTypesAndColumnNames[dataID],
					"tooltipData" : self.getTooltipdata(dataID)}
					
		else:
			return errorMessage
	
	def getPlainColumnNames(self,dataID) -> pd.Index:
		""
		if dataID in self.dfs:
			return self.dfs[dataID].columns
		else:
			return pd.Index

	def getDataDescription(self,dataID : str,columnNames : List[str] | pd.Series | pd.Index):
		""
		return self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"].describe()

	def getDataByColumnNames(self, dataID : str, columnNames : List[str] | pd.Series | pd.Index, rowIdx = None, ignore_clipping : bool = False):
		'''
		Returns sliced self.df
		row idx - boolean list/array like to slice data further.
		'''
		if isinstance(columnNames,pd.Series):
			columnNames = columnNames.values.tolist()
	
		fnComplete = {"fnName":"set_data","fnKwargs":{"data":self.getDataByDataID(dataID,rowIdx,ignore_clipping)[columnNames]}}
		return fnComplete

	def getDataByDataID(self,dataID : str, rowIdx : None | pd.Series | pd.Index = None, ignoreClipping : bool = False) -> pd.DataFrame:
		'''
		Returns the dataframe as a pandas dataset.
		If a clipping is defined (checked by ```hasClipping()```) it returns the clipped dataframe unless ``ìgnore Clipping``is set to True.
		'''

		if dataID not in self.dfs:
			return pd.DataFrame()

		if dataID in self.clippings:
			rowIdx = self.clippings[dataID]
		
		if ignoreClipping or rowIdx is None:
		
			return self.dfs[dataID]
		else:
			return self.dfs[dataID].loc[rowIdx,:]

	def getDataByColumnNameForWebApp(self,dataID : str, columnNames  : str | List[str] | pd.Series | pd.Index) -> dict:
		""
		if isinstance(columnNames,str):
			columnNames = [columnNames]
		if isinstance(columnNames,pd.Index):
			columnNames = columnNames.to_numpy().tolist()
		if isinstance(columnNames,pd.Series):
			columnNames = columnNames.values.tolist() 

		data = self.getDataByColumnNames(dataID,columnNames)["fnKwargs"]["data"]
		#data = data.rename(columns={columnName:"text"})
		data["idx"] = data.index
		return data.to_json(orient="records")

	def getNaNString(self) -> str:
		""
		if not hasattr(self,"replaceObjectNan"):
			return "-"
		return self.replaceObjectNan

	def getNumberUniqueValues(self,dataID : str,columnNames : Iterable) -> Dict[str,int]:
		"""
		"""
		resultDict = OrderedDict()
		if dataID in self.dfs:
			
			for categoricalColumn in columnNames:
				resultDict[categoricalColumn] = self.dfs[dataID][categoricalColumn].unique().size
		
		return resultDict

	def getUniqueValues(self, dataID : str, categoricalColumn : List[str] | str, forceListOutput : bool = False, *args,**kwargs) -> np.ndarray | List[str]:
		'''
		Return unique values of a categorical column. If multiple columns are
		provided in form of a list. It returns a list of pandas series having all
		unique values.
		'''
		if isinstance(categoricalColumn,list):
			if len(categoricalColumn) == 1:
				categoricalColumn = categoricalColumn[0]
				uniqueCategories = self.getDataByDataID(dataID,*args,**kwargs)[categoricalColumn].unique()
			else:
				collectUniqueSeries = []
				X = self.getDataByDataID(dataID,*args,**kwargs)
				for category in categoricalColumn:
					collectUniqueSeries.append(X[category].unique())
				return collectUniqueSeries
		elif isinstance(categoricalColumn,str):
			uniqueCategories = self.getDataByDataID(dataID,*args,**kwargs)[categoricalColumn].unique()
		else:
			return np.array()

		if forceListOutput:
			return [uniqueCategories]
		else:
			return uniqueCategories

	def hasClipping(self,dataID) -> bool:
		""
		return dataID in self.clippings

	def hasData(self) -> bool:
		""
		return len(self.dfs) > 0

	def hasTwoDataSets(self) -> bool:
		""
		return len(self.dfs) > 1

	def renameColumns(self,dataID : str,columnNameMapper : Dict[str,str]) -> dict:
		""
		if dataID in self.dfs:
			columnNameMapper = self.evaluateColumMapper(dataID,columnNameMapper)
			self.dfs[dataID] = self.dfs[dataID].rename(mapper = columnNameMapper, axis = 1)
			#update columns names
			self.extractDataTypeOfColumns(dataID)
			funcProps = getMessageProps("Column renamed.","Column evaluated and renamed.")
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			funcProps["tooltipData"] = self.getTooltipdata(dataID)
			funcProps["columnNameMapper"] = columnNameMapper
			return funcProps
		return getMessageProps("Error..","DataID not found.")

	def setClippingByFilter(self,dataID : str, columnName : str,filterProps : dict, checkedLabels, checkedDataIndex = None):
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
										    "Mask established. Please update graph to see changes.")
				funcProps["maskIndex"] = rowIdxBool
				return funcProps
		
		return getMessageProps("Error..","There was an error when clipping mask was established.")
		
	def evaluateColumnNameOfDf(self, df : pd.DataFrame, dataID : str):
		'''
		Checks each column name individually to avoid same naming and overriding.
		'''
		columns = df.columns.to_numpy()
		evalColumns = []
		#try:
		for columnName in columns:
			evalColumnName = self.evaluateColumnName(columnName,dataID=dataID,extraColumnList=evalColumns)
			evalColumns.append(evalColumnName)
		df.columns = evalColumns
		return df
	
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
				funcProps["encodedColumns"] = columnName
				
				return funcProps
		else:
			return errorMessage

	def getDataValue(self,dataID,columnName,dataIndex,splitString = None) :
		""
		if dataID in self.dfs and columnName in self.dfs[dataID].columns:
			if not isinstance(dataIndex,pd.Series):
				dataIndex = pd.Series(dataIndex)
			values = self.dfs[dataID].loc[dataIndex,columnName].values.flatten()
			if splitString is not None:
				return chain.from_iterable([str(v).split(splitString) for v in values])
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

	def joinDataFrame(self,dataID : str,dataFrame : pd.DataFrame) -> dict:
		"""
		Join a dataframe to a data frame in the collection.
		"""
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
	
	def joinColumnToData(self,dataFrame,dataID,columnNames):
		"Plain return"
		if isinstance(columnNames,str):
			columnNames = [columnNames]

		if dataID in self.dfs:
			columnsToJoin = [colName for colName in columnNames if colName not in dataFrame and colName in self.dfs[dataID].columns]
			columnData = self.dfs[dataID][columnsToJoin]
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
			
			# if bandwidth is None:
			# 	bandwidth = data.index.size**(-1/(data.columns.size+4)) 
			data_noNan = data.dropna()
			if data.shape[1] == 1:
				X = data_noNan.values.flatten()
			else:
				X = data_noNan.values

			hist, bin_edges = np.histogram(X,bins='auto',density=True)
	
			kdeData = np.array([hist[-1] if idx == hist.size else hist[idx] for idx in np.digitize(X,bin_edges,right=True)])
			
			return kdeData, data.index
			print(kdeData)
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
				funcProps["tooltipData"] = self.getTooltipdata(dataID)
				funcProps["dataID"] = dataID
				return funcProps
		else:
			return errorMessage
				
			
         
	def replaceInColumns(self,dataID,findStrings,replaceStrings, specificColumns = None, dataType = "Numeric Floats", mustMatchCompleteCell = False):
		""
		if dataID in self.dfs:
			if specificColumns is None: #replace in columnHeaders
				newColumnNames  = None
				savedColumns = self.dfs[dataID].columns.to_list() 
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
				funcProps["tooltipData"] = self.getTooltipdata(dataID)
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
						for specColumn in specificColumns:
							self.dfs[dataID][specColumn] = self.dfs[dataID][specColumn].astype(str).str.replace(fS, rS,regex=False,case=True)
				else:
					self.dfs[dataID][specificColumns] = self.dfs[dataID][specificColumns].replace(findValues, replaceValues)
				return getMessageProps("Column renamed.","Column evaluated and renamed.")

		else:
			return errorMessage
	
	
	def changeDataType(self, dataID, columnNames, newDataType,flex=False):
		'''
		Changes the DataType of a List of column Names
		'''
		def checkForFloat(row):
			"Try/except to conver entry to float."
			r = np.empty(shape = row.size)
			for n,x in enumerate(row.values):
				try: 
					r[n] = float(x)
				except:
					r[n] = np.nan 
			return r 
			

		if dataID in self.dfs:
			if isinstance(columnNames,pd.Series):
				columnNames = columnNames.values

			try:
				if newDataType == "float64":
					self.dfs[dataID][columnNames] = self.dfs[dataID][columnNames].replace(self.replaceObjectNan,np.nan, regex=False)
				if flex:
					X = self.dfs[dataID][columnNames].apply(checkForFloat,result_type="expand")
					newColumnNames = ["f:{}".format(colName) for colName in columnNames]
					self.joinDataFrame(dataID,pd.DataFrame(X,index=self.dfs[dataID].index, columns=newColumnNames))
				else:
					self.dfs[dataID][columnNames] = self.dfs[dataID][columnNames].astype(newDataType)
			except:
				return getMessageProps("Error..","Changing data type failed.")
			
			if newDataType in ['object','str']:
				self.dfs[dataID][columnNames] = self.dfs[dataID][columnNames].replace("nan",self.replaceObjectNan, regex=False)
				self.dfs[dataID][columnNames].fillna(self.replaceObjectNan,)
				
				#pd.DataFrame().replace()
			#update columns
			self.extractDataTypeOfColumns(dataID)
			funcProps = getMessageProps("Data Type changed.","Columns evaluated and data type changed.")
			funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
			funcProps["tooltipData"] = self.getTooltipdata(dataID)
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

	def copyData(self,dataID):
		""
		if dataID not in self.dfs: return errorMessage

		hasClipping = self.hasClipping(dataID)
		columnNames = self.getPlainColumnNames(dataID)
		#print(columnNames,hasClipping)
		data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
		#print(data)
		originalFileName = self.getFileNameByID(dataID)
		fileName = f"c({originalFileName})" if not hasClipping else f"c(clipping-{originalFileName})"
		dataFrame = data.copy()
		return self.addDataFrame(dataFrame=dataFrame,fileName=fileName)

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
		id = self.getNextDataFrameID()
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
		
	def countQuantProfiles(self,dataID,columnNames):
		"Counts the number of full profiles (e.g. no nans over column names)"
		fileName = self.getFileNameByID(dataID)
		clippingActive = self.hasClipping(dataID)
		data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=False)["fnKwargs"]["data"]
		numberRows = data.index.size
		isNotNullData = ~data.isnull()
		nonNullCounts = isNotNullData.sum(axis=1)
		counts = pd.DataFrame(nonNullCounts.value_counts()).reset_index()
		counts.columns = ["#Valid in out of {}".format(columnNames.size),"#Counts"]
		counts["Rel. Counts ({})".format(numberRows)] = counts["#Counts"] / data.index.size
		return self.addDataFrame(counts,fileName="ProfileCounts({})-Clipping-{}".format(fileName,clippingActive))

	def countValidValuesInColumns(self,dataID,columnNames):
		""
		fileName = self.getFileNameByID(dataID)
		clippingActive = self.hasClipping(dataID)
		data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=False)["fnKwargs"]["data"]
		isNullData = data.isnull()
		
		nans = isNullData.sum(axis=0)
		total = isNullData.count(axis=0)
		valid = total - nans 
		resultMatrix = pd.concat([nans,valid,total],axis=1,ignore_index=True).astype(int).reset_index()
		result = pd.DataFrame(resultMatrix.values,columns=["Column Names","#NaN","#Valid","#Total"])
		result[["#NaN","#Valid","#Total"]] = result[["#NaN","#Valid","#Total"]].astype("int64")
		return self.addDataFrame(result,fileName="Counts({})-Clipping-{}".format(fileName,clippingActive))
	
	def countNaN(self,dataID, columnNames, grouping = None):
		""
		if dataID in self.dfs:
			if grouping is None:
				data = self.dfs[dataID][columnNames].isnull().sum(axis=1)
				return self.addColumnData(dataID,"count(nan):{}".format(mergeListToString(columnNames)),data)
			elif isinstance(grouping,dict):
				grouping = self.parent.grouping.getCurrentGrouping() 
				groupingName = self.parent.grouping.getCurrentGroupingName()
				columnNames = self.parent.grouping.getColumnNames(groupingName)
				X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
				countData = pd.DataFrame(index=X.index, columns = ["count(nan):{}".format(groupName) for groupName in grouping.keys()])
				for groupName, columnNames in grouping.items():
					
					countData["count(nan):{}".format(groupName)] = X[columnNames].isnull().sum(axis=1)
				
				return self.joinDataFrame(dataID,countData)
			else: return getMessageProps("Error..","Grouping must be a dictionary.")
		else:
			return getMessageProps("Error ..","DataID not found.")

	def countValidValues(self,dataID, columnNames, grouping = None):
		"Counts valid values and adds integer columns"
		if dataID in self.dfs:
			if grouping is None:
				
				data = self.dfs[dataID][columnNames].count(axis=1)
				return self.addColumnData(dataID,"count(validV):{}".format(mergeListToString(columnNames)),data)
			else:
				if self.parent.grouping.groupingExists():
					grouping = self.parent.grouping.getCurrentGrouping() 
					groupingName = self.parent.grouping.getCurrentGroupingName()
					columnNames = self.parent.grouping.getColumnNames(groupingName)
					X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
					countData = pd.DataFrame(index=X.index, columns = ["count(validV):{}".format(groupName) for groupName in grouping.keys()])
					for groupName, columnNames in grouping.items():
						
						countData["count(validV):{}".format(groupName)] = X[columnNames].count(axis=1)
					
					return self.joinDataFrame(dataID,countData)
				else:
					return getMessageProps("Error ..","No Grouping found.")

		else:
			return getMessageProps("Error ..","DataID not found.")


	def countValidValuesInSubset(self,dataID,columnNames,categoricalColumns,*args,**kwags):
		""
		if dataID in self.dfs:
			allColumnNames = pd.concat([columnNames,categoricalColumns],ignore_index=True).unique()
			X = self.getDataByColumnNames(dataID,allColumnNames,ignore_clipping=False)["fnKwargs"]["data"]
			countByGroups = X.groupby(by=categoricalColumns).count()
			#rint(countByGroups)
			return self.addDataFrame(countByGroups,fileName="Subset counts") 
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
				removedColumns = columnNames.loc[columnNames.isin(cleanedDf.columns.array)]
				columnNames = [colName for colName in self.getPlainColumnNames(dataID) if colName not in columnNames.values] + cleanedDf.columns.to_list()
				self.dfs[dataID] = self.dfs[dataID][columnNames]
				#update columns names
				self.extractDataTypeOfColumns(dataID)
				funcProps = getMessageProps("Columns removed.","Column evaluated and removed.")
				funcProps["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
				funcProps["columnNames"] = removedColumns
				funcProps["tooltipData"] = self.getTooltipdata(dataID)
				return funcProps
		else:
			return errorMessage
	
		
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
				return getMessageProps("Error ..", "The used fileFormat is unknown.")
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
		if dataID not in self.dfs : return errorMessage 
		try:
			nNaNForDownShift = self.parent.config.getParam("fill.NaN.smart.repl.number.NaN.for.downshift")
			minValidValues = self.parent.config.getParam("fill.NaN.smart.repl.min.number.valid.values")
			downShift = self.parent.config.getParam("fill.NaN.gaussian.downshift")
			widthGaussian = self.parent.config.getParam("fill.NaN.gaussian.width")
			if all(x is not None for x in [nNaNForDownShift,minValidValues,downShift,widthGaussian]):
				smartRrep = ICSmartReplace(
						nNaNForDownshift=nNaNForDownShift,
						minValidvalues=minValidValues,
						grouping=grouping,
						downshift=downShift,
						scaledWidth=widthGaussian)
						
				data = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
				X = smartRrep.fitTransform(data)
			
				self.dfs[dataID].loc[X.index,X.columns] = X
				return getMessageProps("Done ..","Replacement done.")

		except Exception as e:
			
			return getMessageProps("Error ..","There was an error: "+str(e))

	def formatStringColumn(self, dataID : str, columnNames : pd.Index | pd.Series, formatFn : str) -> dict:
		""""""
		if dataID not in self.dfs : return errorMessage 
		X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
		newColumnNames = [f"({formatFn}){colName}" for colName in columnNames]
		Y = pd.DataFrame(columns=newColumnNames, index=X.index)
		for n, colName in enumerate(columnNames):
			newColumnName  = newColumnNames[n]
			if formatFn == "upper":
				Y.loc[X.index,newColumnName] = X[colName].str.upper() 
			elif formatFn == "lower":
				Y.loc[X.index,newColumnName] = X[colName].str.lower() 
			elif formatFn == "capitilize":
				Y.loc[X.index,newColumnName] = X[colName].str.capitalize() 
		return self.joinDataFrame(dataID,Y)

	def summarizeGroups(self,dataID,grouping,metric,**kwargs):
		""
		if dataID not in self.dfs : return errorMessage 
		for groupName, columnNames in grouping.items():

			data = self.transformer.summarizeTransformation(dataID,columnNames,metric=metric,justValues = True)
			columnName = "s:{}({})".format(metric,groupName)
			self.addColumnData(dataID,columnName,data)

		completeKwargs = getMessageProps("Done..","Groups summarized. Columns added.")
		completeKwargs["columnNamesByType"] = self.dfsDataTypesAndColumnNames[dataID]
		completeKwargs["tooltipData"] = self.getTooltipdata(dataID)
		return completeKwargs
		

	def removeDuplicates(self,dataID : str, columnNames : pd.Series):
		""
		if dataID in self.dfs:
			df = self.dfs[dataID].drop_duplicates(subset=columnNames.values)
			fileName = self.getFileNameByID(dataID)
			return self.addDataFrame(df,fileName="dropDup:({})".format(fileName))
		else:
			return errorMessage

	def replaceSelectionOutlierWithNaN(self,dataID : str,columnNames : pd.Series):
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
				return getMessageProps("Error", "There was an error: "+str(e))

	def replaceGroupOutlierWithNaN(self,dataID : str, grouping : dict):
		""
		if dataID in self.dfs:
			
				m = self.parent.config.getParam("outlier.iqr.multiply")
				copy = self.parent.config.getParam("outlier.copy.results")
				if copy: 
					cleanColumns = []
					groupingColumnNames = self.parent.grouping.getColumnNames()
					numColumns = len(groupingColumnNames)
					cleanColumns = [f"reO({colName})" for colName in groupingColumnNames]

					cleanData = pd.DataFrame(
						np.zeros(shape=(self.dfs[dataID].index.size,numColumns)), 
			      				index=self.dfs[dataID].index, columns = cleanColumns).astype(bool)

				for n, (groupName, columnNames) in enumerate(grouping.items()):
					cleanGroupColumns = [f"reO({colName})" for colName in columnNames]
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
						cleanData[cleanGroupColumns] = X 

				if not copy:
					return getMessageProps("Done ..","Outlier replaced with NaN.")
				else:
					return self.joinDataFrame(dataID,cleanData)
		else:
			return errorMessage

	def fillNaNBy(self,dataID,columnNames,fillBy = "Row mean"):
		""
		X = self.getDataByColumnNames(dataID,columnNames,ignore_clipping=True)["fnKwargs"]["data"]
		#create matrix with bools where nans are located.
		nanBoolIdx = X.isna().replace({True:"+",False:self.replaceObjectNan})
		nanBoolIdx.columns = [f"imp:{colName}" for colName in X.columns]
		if isinstance(fillBy,float):
			self.dfs[dataID].loc[X.index,X.columns] = X.fillna(fillBy)
		elif isinstance(fillBy,str):
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
			elif fillBy == "Gaussian distribution":
				downShift = self.parent.config.getParam("fill.NaN.gaussian.downshift")
				widthGaussian = self.parent.config.getParam("fill.NaN.gaussian.width")
				boolIdx = X.isna()
				for columnName in columnNames:
					columnBoolIdx = boolIdx[columnName].values.flatten()
					vvs = X.loc[:,columnName].values.flatten() #values
					avg = np.nanmean(vvs)
					std = np.nanstd(vvs)
					dwnShift = np.random.normal(loc=avg-downShift * std, scale = std * widthGaussian, size = (vvs.size,))
					X.loc[columnBoolIdx,columnName] = dwnShift[columnBoolIdx]
				self.dfs[dataID].loc[X.index,X.columns] = X

			else:
				return getMessageProps("Error..","FillBy method not found.")
		else:
			return getMessageProps("Error..","FillBy method not found.")
		
		kwargs = {**self.joinDataFrame(dataID,nanBoolIdx),**getMessageProps("Done ..","NaN were replaced.")}
		return kwargs
	

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
	
	
	def getCategoricalColumns(self,dataID : str) -> List[str]:
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
		Returns columns names that are floats
		'''
		if dataID in self.dfs:
			return self.dfsDataTypesAndColumnNames[dataID]['Numeric Floats']

		return []

	def getIntegerColumns(self,dataID):
		'''
		Returns columns names that are integers
		'''
		if dataID in self.dfs:
			return self.dfsDataTypesAndColumnNames[dataID]['Integers']

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
			if selectedColumns is None:
				X = self.dfs[dataID][list(filterIdx.keys())]
				for columnName, idx in filterIdx.items():
					X.loc[idx,columnName] = np.nan
				X.columns = ["{}::{}".format(baseString,colName) for colName in X.columns]
			elif isinstance(selectedColumns,dict):
				totalColumns = np.unique(list(selectedColumns.values())).tolist()
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
	
	def getFileNames(self) -> List[str]:
		'''
		Returns the available file names
		'''
		return list(self.fileNameByID.values())

	def getDataIDbyFileNameIndex(self,idx):
		""
		return [dataID for n,dataID in enumerate(self.fileNameByID.keys()) if n in idx]

	def setFileNameByID(self,dataID : str ,fileName : str) -> dict:
		"Renames the dataframe for the user, id is unchanged"
		if dataID in self.fileNameByID:
			self.fileNameByID[dataID] = fileName	
			completeKwargs = getMessageProps("Renamed.","Data frame renamed.")
			completeKwargs["dfs"] = self.fileNameByID
			completeKwargs["remainLastSelection"] = True
			return completeKwargs
		return getMessageProps("Error..","DataID did not match any of the loaded data.")
	
	def getFileNameByID(self,dataID : str) -> str:
		""
		if dataID in self.fileNameByID:
			return self.fileNameByID[dataID]
	
	def getNextDataFrameID(self):
		'''
		To provide consistent labeling, use this function to get the id the new df should be added
		'''
		self.dataFrameId += 1
		idForNextDataFrame = 'DataFrame: {}'.format(self.dataFrameId)
		
		return idForNextDataFrame
	
	def getRowNumber(self,dataID : str) -> int:
		'''
		Returns the number of rows.
		'''
		if dataID in self.dfs:
			return self.dfs[dataID].index.size
		return 	0
	
	def getIndex(self,dataID) -> pd.Index:
		'''
		Returns the number of rows.
		'''
		if dataID in self.dfs:
			return self.dfs[dataID].index
		return 	pd.Index()
	
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
		
	def meltData(self,dataID : str, columnNames : pd.Series) -> dict:
		'''
		Melts data frame.
		'''
		if dataID in self.dfs:
			addColumnNames = self.parent.config.getParam("melt.data.add.column.names")
			ignoreClipping = self.parent.config.getParam("melt.data.ignore.clipping")
			idVars = [column for column in self.dfs[dataID].columns if column not in columnNames.values] #all columns but the selected ones
			potentialValueName = "melt_value" if not addColumnNames else 'melt_value:{}'.format(mergeListToString(columnNames)).replace("'",'')
			potentialVariableName = "melt_variable" if not addColumnNames else 'melt_variable:{}'.format(mergeListToString(columnNames)).replace("'",'')
			valueName = self.evaluateColumnName(columnName = potentialValueName, dataID = dataID)
			variableName = self.evaluateColumnName(columnName=potentialVariableName, dataID = dataID)

			if self.hasClipping(dataID) and not ignoreClipping:
				rowIdx = self.clippings[dataID]
				df = self.dfs[dataID].loc[rowIdx,:]
			else:
				df = self.dfs[dataID]

			meltedDataFrame = pd.melt(df, id_vars = idVars, value_vars = columnNames,
									var_name = variableName,
									value_name = valueName)
			#add groupings
			if self.parent.grouping.groupingExists():
				groupingNames = self.parent.grouping.getGroupings()
				for groupingName in groupingNames:
					mapper = self.parent.grouping.getGroupNameByColumn(groupingName)
					meltedDataFrame.loc[:,groupingName] = meltedDataFrame[variableName].map(mapper).fillna(self.replaceObjectNan)

			## determine file name
			baseFile = self.getFileNameByID(dataID)
			numMeltedfiles = len([fileName for fileName in self.fileNameByID.values() if 'Melted_' in fileName])			
			fileName =  'melt({})_{}'.format(baseFile,numMeltedfiles)	

			return self.addDataFrame(meltedDataFrame, fileName=fileName)					
		else:
			return errorMessage


	def correlateDfs(self,corrParams : dict) -> dict:
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

	def correlateFeaturesDfs(self,corrParams : dict) -> dict:
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

	def mergeDfs(self, mergeParams : dict, how : str = "left", indicator : bool = True) -> dict:
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

			try:
				mergedDataFrames = leftDf.merge(rightDf,
								how = how, 
								left_on = leftMergeColumn, 
								right_on = rightMergeColumn, 
								indicator = indicator)
			except Exception as e:
				return getMessageProps("Error...","Calling merge resulted in an error. Please make sure to use the same datatype for key columns. " + str(e))
			mergeIndicatorColumns = [colName for colName in mergedDataFrames.columns if "_merge" in colName]
			if len(mergeIndicatorColumns) > 0:
				mergedDataFrames[mergeIndicatorColumns] = mergedDataFrames[mergeIndicatorColumns].astype(str)
			mergedDataFrames.reset_index(drop=True,inplace=True)
			return self.addDataFrame(
							dataFrame = mergedDataFrames, 
							fileName = "merged({}:{})".format(self.fileNameByID[leftDataID],self.fileNameByID[rightDataID]),
							cleanObjectColumns=True
							)

		return errorMessage



	def pivotTable(self,dataID, indexColumn, columnNames, findCommonString = True):
		""
		if dataID in self.dfs:
			if findCommonString:
				commonString = findCommonStart(*self.getUniqueValues(dataID,columnNames))
			else:
				commonString = ""
			data = pd.pivot_table(data = self.dfs[dataID], columns=[columnNames], index=[indexColumn])
			mergedColumns = ['_'.join(col).strip().replace(commonString,"") for col in data.columns.array]
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
		
	def transposeDataFrame(self,dataID : str, columnNames = None, columnLabel = None) -> dict:
		""
		if dataID in self.dfs:
			df = self.dfs[dataID]
			newColumnNames = df[columnLabel].values.flatten()

			if np.unique(newColumnNames).size != df.shape[0]:
				newColumnNames = ["{}_{}".format(newColumnNames[n],n) for n in np.arange(newColumnNames.size)]

			if columnLabel is not None:
				if columnNames is not None:
					requiredColumnNames = [colName for colName in df.columns if colName != columnLabel and colName in columnNames.values]
				else:
					requiredColumnNames = [colName for colName in df.columns if colName != columnLabel]
			else:
				requiredColumnNames = df.columns.array 
	
			dataT = df[requiredColumnNames].T 
			dataT.columns = newColumnNames

			dataT = dataT.reset_index()

			return self.addDataFrame(dataT,fileName="t:{}".format(self.getFileNameByID(dataID)))

		return errorMessage
	

	def updateData(self,dataID : str, data : pd.DataFrame) -> dict:
		'''
		Updates dataframe, input: dataID and data
		'''
		if dataID in self.dfs:
			self.dfs[dataID] = data
			self.extractDataTypeOfColumns(dataID)
			
			return getMessageProps("Updated..","Data ({}) updated.".format(dataID))
		else:
			return errorMessage
		
			
	def sortData(self, dataID : str, columnNames : pd.Series, kind : str= 'mergesort', ascending : bool = True, na_position : str = 'last') -> dict:
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
			completeKwargs["tooltipData"] = self.getTooltipdata(dataID)
			return completeKwargs
		else:
			return errorMessage

			
	def saveFileName(self,dataID,fileName):
		'''
		'''
		self.fileNameByID[dataID] = fileName
						

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

		
		
			
		
				
		
	
		
		
		
	
	
	
	
	
	
		
		










