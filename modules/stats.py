"""
	""CLASS TO HANDLE STATISTICAL TESTS""
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



import numpy as np
import pandas as pd
import itertools

from scipy.stats import linregress
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ranksums
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
from scipy.stats import f_oneway
from scipy import interpolate


from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.decomposition import PCA, IncrementalPCA, NMF, TruncatedSVD, FactorAnalysis
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from collections import OrderedDict

from modules.calculations.anova import Anova
from modules.utils import *
from modules.calculations.pls_da import PLS_DA

dimensionalReductionMethods = OrderedDict([('Principal Component Analysis',PCA),
								('Non-Negative Matrix Factorization',NMF),
								('Incremental PCA',IncrementalPCA),
								#('Latent Semantic Analysis',TruncatedSVD),
								#('t-distributed Stochastic Neighbor Embedding',TSNE),
								#('Linear Discriminant Analysis',LinearDiscriminantAnalysis),
								('Factor Analysis',FactorAnalysis),
								])
#('PLS-DA - Partial Least Squares - Discrimant Analysis',PLS_DA)
dic = {'t-distributed Stochastic Neighbor Embedding':{'init':'pca','perplexity':30,
		'learning_rate':100,'method':'barnes_hut','early_exaggeration':3}}

class dimensionReductionCollection(object):
	'''
	Little helper class to save results of a dimensional
	reduction method. 
	
	Comment - this might be the wrong place for this class
	will probably be moved soon somewhere else .. 
	'''
	def __init__(self):
		self.id = 1
		self.set_max_comps()
		self.dimRedResults = OrderedDict()
		
	def save_calculation(self,resultDict,numericColumns,method, dfID):
		'''
		'''
		self.id += 1
		
		results = {'data':resultDict,
				   'numericColumns':numericColumns,
				   'method':method,
				   'dataID':dfID,
				   'name':'{}_{}'.format(method,self.id)}
				   
		self.dimRedResults[self.id] = results
	
	def get_last_calculation(self):
		'''
		'''
		return self.dimRedResults[self.id]
	
	def get_calculation(self,id):
		'''
		'''
		return self.dimRedResults[id]
	
	def set_max_comps(self,comps = 3):
		'''
		'''
		self.nComponents = comps
	
	def get_drivers_and_components(self, id = None, which = 'Both'):
		'''
		Which can be either 'Both','Drivers', or 'Components'
		'''
		if which not in ['Both','Drivers','Components']:
			return
		drivers, components = None, None
		if id is None:
			id = self.id
		
		if which in ['Both','Drivers']:
			drivers = self.dimRedResults[id]['data']['Drivers']
			
		if which in ['Both','Components']:
			components = self.dimRedResults[id]['data']['Components']
		
		
		return drivers, components, \
		self.dimRedResults[id]['numericColumns'], \
		self.dimRedResults[id]['dataID']

	def update_dict(self,key,value):
		'''
		Function that can be used to update the result dict. 
		For example when highlighting them by color.
		'''
		base = self.dimRedResults[self.id]
		base[key] = value
		self.dimRedResults[self.id] = base
		
		
def get_dimensionalReduction_results(dataFrame, nComps = None, method = 'PCA', outcomeVariable = None):
	'''
	dataFrame might be either a pandas dataframe or directly a ndarray.
	'''
	outputDict = dict() 
	
	if isinstance(dataFrame, pd.DataFrame):
		columnsNames = dataFrame.columns.values.tolist()
		if nComps is None:
			nComps = len(columnsNames)
		dataFrame.dropna(inplace=True) 
		
		data = dataFrame.as_matrix()
	
	if method == 'Latent Semantic Analysis':
		if nComps is None:
			nComps -= 1
	
	

	kwargs = dic[method] if method in dic else {}
	dimReductionClass = dimensionalReductionMethods[method](n_components = nComps,**kwargs)
	
	if method == 't-distributed Stochastic Neighbor Embedding':
		drivers = dimReductionClass.fit_transform(data)
	elif method == 'Linear Discriminant Analysis':
		dimReductionClass.fit(data,outcomeVariable)
		drivers = dimReductionClass.transform(data)
		#print(drivers)
		#print(dimReductionClass.get_params())
		coeffs = dimReductionClass.coef_
		outputDict['Components'] = pd.DataFrame(coeffs, columns = columnsNames,
										index = ['Comp_'+str(i+1) for i in range(coeffs.shape[0])])
		
		#print(drivers.shape)
		#print(dimReductionClass.predict_proba(data))
		#components = dimReductionClass.components_
		
	else:
		dimReductionClass.fit(data)
		components = dimReductionClass.components_
		drivers = dimReductionClass.transform(data)
	
		outputDict['Components'] = pd.DataFrame(components, columns = columnsNames,
										index = ['Comp_'+str(i+1) for i in range(components.shape[0])])
	if method == 't-distributed Stochastic Neighbor Embedding':
		outputDict['klDivergence'] = dimReductionClass.kl_divergence_	
	elif method == 'Factor Analysis':
		outputDict['noiseVariance'] = dimReductionClass.noise_variance_							
	elif method != 'Non-Negative Matrix Factorization':
		outputDict['ExplainedVariance'] = pd.DataFrame(dimReductionClass.explained_variance_ratio_)
	else:
		outputDict['ReconstructionError'] = dimReductionClass.reconstruction_err_
		
	outputDict['Drivers'] = pd.DataFrame(drivers, index = dataFrame.index, 
		columns = ['Comp_'+str(i+1) for i in range(drivers.shape[1])])
	outputDict['Predictor'] = dimReductionClass
	
	return outputDict
	

def get_linear_regression(dataFrame):
	'''
	dataFrame - pandas 
	'''
	dataFrame.dropna(inplace=True)
	x = dataFrame.iloc[:,0].values
	y = dataFrame.iloc[:,1].values
	
	slope, intercept, r_value, p_value, std_err = linregress(x,y)
	x1, x2 = x.min(), x.max()
	y1, y2 = slope*x1+intercept, slope*x2+intercept
	
	return [x1,x2],[y1,y2],slope, intercept, r_value, p_value, std_err
	
    
def get_lowess(dataFrame):
	'''
	Calculates lowess line from dataFrame input
	'''
	dataFrame.dropna(inplace=True)
	x = dataFrame.iloc[:,0].values
	y = dataFrame.iloc[:,1].values
	
	lenX = x.size
	if lenX > 1000:
		it = 3
		frac = 0.65
	else:
		it = 1
		frac = 0.3
		
	lowessLine = lowess(y,x, it=it, frac=frac)
	
	return lowessLine



def calculate_anova(dataframe, dependentVariable, wFactors= [], bFactors = [],
					subjectColumn = 'SUBJECT'):
	'''
	Calculate ANOVA for given Factors. 
	
	Parameters:
	=============
	data frame: pandas dataframe
	dependentVariable : string indicating column in dataframe (must be integer or float)
	wFactors : list of strings or string indicating columns for within subject calculations
	bFactors : list of strings or string indicating columns for between subject calculations
	subjectColumns : string indicating where subject numbers (integers) are found. This 
					 also will be specified, when no within factors are given.
	'''
	
	if isinstance(wFactors,str):
		wFactors = [wFactors]
	if isinstance(bFactors,str):
		bFactors = [bFactors]
	
	factors = wFactors + bFactors 
	
	if subjectColumn not in dataframe.columns:
		if len(wFactors) != 0:
			return
		else:
			dataframe.loc[:,'SUBJECT'] = np.arange(1,len(dataframe.index)+1)
			subjectColumn = 'SUBJECT'
	
	if (len(wFactors) == 0 and len(bFactors) != 0) or \
	(len(wFactors) != 0 and len(bFactors) == 0):
		
		aov =  Anova(dataframe, dependentVariable, wFactors= wFactors, bFactors = bFactors,
					subjectColumn = subjectColumn)
		
	elif len(wFactors) != 0 and len(bFactors) != 0:
	
		aov = Anova(dataframe, dependentVariable, 
												  wFactors= wFactors, bFactors = bFactors,
												  subjectColumn = subjectColumn)
	result = aov.finalResult
	title = aov.title
	del aov
	return result, title 




def round_on_resolution(value,precision):
	'''
	
	Returns a rounded value on given precision. 
	
	Parameters:
	==============
	value: float number
	precision: float number
	'''
	return round(value/precision) * precision


class statisticResultCollection(object):
	'''
	'''
	def __init__(self):
		'''
		'''
		self.displayStatsPerformed = pd.DataFrame(columns = ['index','group1','group2','test settings',
															'p-value','test-statistic'])
		
	def save_test(self, df):
		'''
		'''
		self.displayStatsPerformed = self.displayStatsPerformed.append(df, ignore_index = True)
		
	
	@property
	def performedTests(self):
		'''
		'''
		return self.displayStatsPerformed
		
		



	

class interactiveStatistics(object):
	'''
	'''
	def __init__(self,plotter,dfClass,selectedTest,resultCollection):
		'''
		'''
		self.subplotId = None
		self.saveStatsPerformed = OrderedDict()
		self.displayStatsPerformed = pd.DataFrame(columns = ['index','group1','group2','test settings',
															'p-value','test-statistic'])
		
		self.resultCollection = resultCollection
		self.saveClickCoords = dict() 
		self.textPValues = []
		self.selectedTest = selectedTest
				
		self.plotter = plotter
		self.dataId = self.plotter.get_dataID_used_for_last_chart()
		self.plotHelper = self.plotter.get_active_helper()
		self.numericColumns, self.categoricalColumns = self.plotHelper.columns
		self.numbCategoricalColumns = len(self.categoricalColumns)
		
		self.dfClass = dfClass
		self.dfClass.set_current_data_by_id(id = self.dataId)
		
		self.compareTwoGroups = self.plotter.figure.canvas.mpl_connect('button_press_event',  lambda event:
														self.identify_test_groups_and_perform_test(event))
														
	def disconnect_event(self):
		'''
		Disconnects click event and saves data to plotter 
		'''
		self.plotter.figure.canvas.mpl_disconnect(self.compareTwoGroups)
		self.plotter.save_statistics(self.saveStatsPerformed,self.displayStatsPerformed)
		try:
			self.disconnect_moving_events()
		except:
			pass				
											
	def reset_identification_process(self):
		'''
		'''
		self.saveClickCoords.clear()
		
	def make_text_movable(self,key,paramDict):
		
		yLimOrig = paramDict['axis'].get_ylim()
		
		self.onMotionHandler = self.plotter.figure.canvas.mpl_connect('motion_notify_event',
													lambda event: self.on_motion(event,key,paramDict,yLimOrig))
		self.onRealeaseHandler = self.plotter.figure.canvas.mpl_connect('button_release_event',
													lambda event: self.on_release(event,key,paramDict))
													
													
	
	def on_motion(self,event,key,paramDict,yLimOrig):
		'''
		'''
		if event.inaxes is None:
			self.disconnect_moving_events()
			return
			
		paramDict['text'].set_position((event.xdata,event.ydata))
		yDataLine = paramDict['line'].get_ydata()
		yDataLine[[1,2]] = event.ydata
		paramDict['line'].set_ydata(yDataLine)
		
		ax = event.inaxes
	
		yAxisMin, yAxisMax = ax.get_ylim()
		
		if event.ydata > yAxisMax - 0.08* yAxisMax:
			ax.set_ylim([yAxisMin, yAxisMax+yAxisMax*0.08])
		elif event.ydata < yAxisMax - 0.06+yAxisMax:
			if yAxisMax < yLimOrig[1]:
				pass
			else:
				ax.set_ylim([yAxisMin, yAxisMax-yAxisMax*0.04])
	
			
		self.plotter.redraw()
		
	def on_release(self,event,key,paramDict):
		'''
		'''
		paramDict['linesAndText']['y'] = paramDict['line'].get_ydata()
		paramDict['linesAndText']['textPos'] = paramDict['text'].get_position()
		self.saveStatsPerformed[key] = paramDict
		
		self.disconnect_moving_events()
		
	@property
	def performedTests(self):
		'''
		Following property/attribute naming style (no "_"). Right now the code 
		is sometimes inconsistent in that naming style. Working on it ..
		'''
		return self.displayStatsPerformed
		
	def disconnect_moving_events(self):
		'''
		'''	
		self.plotter.figure.canvas.mpl_disconnect(self.onMotionHandler)
		self.plotter.figure.canvas.mpl_disconnect(self.onRealeaseHandler)
		
	def delete_all_stats(self):
		'''
		'''
		for key,param in self.saveStatsPerformed.items():
			self.delete_text_and_line(key,param,delAndReplot = False)
			
		self.saveStatsPerformed.clear()
		
		
	def delete_text_and_line(self,key,param, delAndReplot = True):
		'''
		'''
		param['text'].remove()
		param['line'].remove()
		if delAndReplot:
			del self.saveStatsPerformed[key] 
			self.plotter.redraw()
			
	def identify_test_groups_and_perform_test(self,event):
		'''
		'''
		self.plotter.castMenu = True
		for id,param in self.saveStatsPerformed.items():
			if param['text'].contains(event)[0]:
				self.plotter.castMenu = False
				if event.button == 1:
					self.make_text_movable(id,param)
				elif event.button in [2,3]:
					self.delete_text_and_line(id,param)
					
				return
				
		if event.button != 1:
			return
			
		if event.inaxes is None:
			return
			
		xDataEvent = event.xdata
		if self.numbCategoricalColumns == 0:
			indexClick = round(xDataEvent,0)
		else:
			indexClick = round_on_resolution(xDataEvent,0.02)
			
		## save click 1 and 2 
		if 'indexClick1' not in self.saveClickCoords:
			self.saveClickCoords['indexClick1'] = indexClick
			self.saveClickCoords['xyCoordsClick1'] = (xDataEvent,event.ydata)
			self.saveClickCoords['axis'] = event.inaxes
			self.visualize_first_click()
			return
		
			
		elif 'indexClick2' not in self.saveClickCoords:
			self.remove_first_click()
			if indexClick == self.saveClickCoords['indexClick1']:
				tk.messagebox.showinfo('Same group ..','Same data selected. Resetting ..')
				self.reset_identification_process()
				return
				
			if event.inaxes != self.saveClickCoords['axis']:
				tk.messagebox.showinfo('Error ..','Statistical testing in interactive mode is only'+
								 ' possible within a subplot. You can use "Get all combinations"'+
								 ' from the misc menu to compare all groups pairwise. Resetting ..')
				self.reset_identification_process()
				return
			
			self.saveClickCoords['indexClick2'] = indexClick
			self.saveClickCoords['xyCoordsClick2'] = (xDataEvent,event.ydata)
			
						
			groupList = self.get_groups()
			testResult = self.perform_test(groupList)
			
			self.saveClickCoords['testResult'] = testResult
			
			self.plot_sigLines_and_pValue()


	def remove_first_click(self, redraw = False):
		'''
		Removes cross to indicate clicked group
		'''
		if hasattr(self,"scatFirstClick") and self.scatFirstClick is not None:
			self.scatFirstClick.remove()
			if redraw:
				self.plotter.redraw()		

	def visualize_first_click(self):
		'''
		Adds red cross to indicated the selected group
		'''
		self.scatFirstClick = self.saveClickCoords['axis'].scatter(
													self.saveClickCoords['indexClick1'],
													self.saveClickCoords['xyCoordsClick1'][1],
													c = "red", marker = 'X',
													)
		self.plotter.redraw()					
		
	def get_groups(self):
		'''
		'''
		## make sure that correct dataframe is selected
		self.dfClass.set_current_data_by_id(id = self.dataId)
		numbNumericColumns = len(self.numericColumns)
		testGroups = []
		idxKeys = ['indexClick1','indexClick2']
		if self.numbCategoricalColumns == 0:
		
			for n,idxKey in enumerate(idxKeys):
				idx = int(round(self.saveClickCoords[idxKey],0))
				if idx == numbNumericColumns: ## if user clicked to far to the right
					idx = -1
				columnData = self.numericColumns[idx]
				self.saveClickCoords['group{}'.format(n+1)] = [columnData]
				testGroups.append(self.dfClass.df[columnData].values)
				 		
		elif self.plotter.splitCategories == False:
			
			positionsInPlot = np.linspace(0,self.numbCategoricalColumns,num=self.numbCategoricalColumns+1)
			subplotNum = self.plotter.get_number_of_axis(self.saveClickCoords['axis'])
			numericColumn = self.numericColumns[subplotNum]
			for n, idxKey in enumerate(idxKeys):
				idx = int(round(self.saveClickCoords[idxKey],0))
				# +1 because we have one category "Whole population" extra
				if idx >= self.numbCategoricalColumns+1:
					idx -= 1
				elif idx < 0:
					idx = 0
				
				## overwrite exact xPosition to have algined singificance lines								   			
				self.saveClickCoords[idxKey] = idx	
				
				columnData, categoricalColumn = self.get_data_if_no_split(idx,numericColumn)
				groupName = '{}({})'.format(categoricalColumn,numericColumn,positionsInPlot)
				self.saveClickCoords['group{}'.format(n+1)] = [groupName]	
				testGroups.append(columnData)		
					
		elif self.numbCategoricalColumns == 1:
			
			categoricalValues = self.dfClass.get_unique_values(self.categoricalColumns[0])
			numCategories = categoricalValues.size
			possibleCombinations = list(itertools.product(self.numericColumns,categoricalValues))
			if numbNumericColumns > 1:
			
				offsetPerCategory = 0.4-(0.8/numCategories)/2 
				positionsInPlot = np.linspace(-offsetPerCategory , offsetPerCategory ,
												num = numCategories)
			else:
				positionsInPlot = np.linspace(0,numCategories-1,num=numCategories)
				
			groupedData = self.dfClass.get_groups_by_column_list(self.categoricalColumns)
			testGroups = self.identify_and_subset_groups(positionsInPlot,possibleCombinations,
														 numbNumericColumns,groupedData) 
		
		elif self.numbCategoricalColumns >= 2:
			
			nAxis = self.plotter.get_number_of_axis(self.saveClickCoords['axis'])
			if self.numbCategoricalColumns == 2:
			
				numericalColumn = self.numericColumns[nAxis]
				groupedData = self.dfClass.get_groups_by_column_list(self.categoricalColumns[:2])
				
			elif self.numbCategoricalColumns == 3:
			
				categoricalLevel3 = self.categoricalColumns[2]
				uniqueLevel3 = self.dfClass.get_unique_values(categoricalLevel3).tolist()
				sizeUniqueLevel3 = len(uniqueLevel3) # this is exactly the number of subplots
				uniqueLevel3 = uniqueLevel3*sizeUniqueLevel3
				numericColumnIdx = int(nAxis/sizeUniqueLevel3)
				numericalColumn = self.numericColumns[numericColumnIdx]
				groupedDataLevel3 = self.dfClass.get_groups_by_column_list([self.categoricalColumns[2]])
				groupName = uniqueLevel3[nAxis]
				self.saveClickCoords['dataSubset'] = groupName
				groupLevel3 = groupedDataLevel3.get_group(groupName)
				groupedData = groupLevel3.groupby(self.categoricalColumns[:2], sort=False)
				
				
			self.saveClickCoords['nAxis'] = nAxis
			
			categoricalLevel1, categoricalLevel2 = self.categoricalColumns[:2]
			uniqueLevel1, uniqueLevel2 = self.dfClass.get_unique_values(self.categoricalColumns[:2])
			possibleCombinations = list(itertools.product(uniqueLevel1,uniqueLevel2))
			nLevel2 = uniqueLevel2.size
			offsetPerCategory = 0.4-(0.8/nLevel2)/2 
			positionsInPlot = np.linspace(-offsetPerCategory , offsetPerCategory ,
												num = nLevel2)
			
			
			testGroups = self.identify_and_subset_groups(positionsInPlot,possibleCombinations,
											uniqueLevel1.size,groupedData,
											numbCategoricalColumns = self.numbCategoricalColumns,
											numericalColumnForTest = numericalColumn)
			
		return testGroups	

	def get_data_if_no_split(self,idx,numericColumn):
		'''
		This works only if "+" is found in data. 
		'''
		if idx != 0:
			categoricalColumn = self.categoricalColumns[idx-1]
		else:
			categoricalColumn = 'Whole population'
					
		idString = '{}_{}_{}'.format(idx,numericColumn,categoricalColumn)
		# check if dict to save data exists, if not create
		if hasattr(self,'savedData') == False:
			self.savedData = dict() 
		# if subsetting was performed already, just return data
		elif idString in self.savedData:
			return self.savedData[idString], categoricalColumn
		
		if idx != 0:
			boolIndicator = self.dfClass.df[categoricalColumn].str.contains('^\+$')
			data = self.dfClass.df.loc[boolIndicator,numericColumn].values
		else:
			data = self.dfClass.df[numericColumn].values
		#save data to avoid re subsetting if used again
		self.savedData[idString]  = data 
		return data, categoricalColumn
		
		
		
		
		
		

	def identify_and_subset_groups(self,positionsInPlot, possibleCombinations, numSeparationsInPlot, groupedData, 
									numbCategoricalColumns = 1, numericalColumnForTest = None):
		'''
		'''
		
		for levelIndex in range(numSeparationsInPlot):
				if levelIndex == 0:
					allPossiblePositions = positionsInPlot
				else:
					addLevelPostions = allPossiblePositions + levelIndex
					allPossiblePositions = np.append(allPossiblePositions,addLevelPostions)
			
		idxClick1, idxClick2 = \
					find_nearest_index(allPossiblePositions,self.saveClickCoords['indexClick1']),\
					find_nearest_index(allPossiblePositions,self.saveClickCoords['indexClick2'])
		groupNameClick1, groupNameClick2 = tuple(possibleCombinations[idxClick1]), \
											   tuple(possibleCombinations[idxClick2])
		## overwrite exact xPosition to have algined singificance lines								   			
		self.saveClickCoords['indexClick1'], self.saveClickCoords['indexClick2'] = \
												allPossiblePositions[idxClick1], allPossiblePositions[idxClick2]			   
		if numbCategoricalColumns == 1:
				testGroups = [groupedData.get_group(groupNameClick1[1])[groupNameClick1[0]].values,
						  groupedData.get_group(groupNameClick2[1])[groupNameClick2[0]].values]
						  
				self.saveClickCoords['group1'] = groupNameClick1
				self.saveClickCoords['group2'] = groupNameClick2
		else:
				testGroups = [groupedData.get_group(groupNameClick1)[numericalColumnForTest].values,
						  groupedData.get_group(groupNameClick2)[numericalColumnForTest].values]	
				self.saveClickCoords['group1'] = groupNameClick1 + (numericalColumnForTest,)
				self.saveClickCoords['group2'] = groupNameClick2 + (numericalColumnForTest,)
		return testGroups			
		

	def perform_test(self, groupList):
		'''
		'''
		if any(group.size < 2 for group in groupList):
			tk.messagebox.showinfo('Error ..','Less than 2 observations in one of the selected groups. Aborting ..')
			
		elif any(group.size < 4 for group in groupList):
			tk.messagebox.showinfo('Warning ..','Less than 4 observations in one of the selected groups.')
		
		testResult = compare_two_groups(self.selectedTest,groupList)
		return testResult
			
	def save_test(self,paramDict):
		'''
		'''
		id = len(self.saveStatsPerformed)
		self.saveStatsPerformed[id] = paramDict.copy()
		
		collectData = dict()
		testResult = paramDict['testResult']
		for column in self.displayStatsPerformed.columns:
			if column == 'index':
				collectData[column] = id
			elif column in ['group1','group2']:
				groupName = get_elements_from_list_as_string(paramDict[column], maxStringLength = None)
				if 'dataSubset' in paramDict:
					groupName = '{}, {}'.format(paramDict['dataSubset'],groupName)
				collectData[column] = groupName
			elif column == 'test settings':
				collectData[column] = str(self.selectedTest).replace('{','').replace('}','').replace("'",'')
			elif column  == 'p-value':
				collectData[column] = testResult[1]
			elif column == 'test-statistic':
				collectData[column] = testResult[0]
				
		dfResult = pd.DataFrame(collectData, columns = self.displayStatsPerformed.columns, index = [0])		
		self.displayStatsPerformed = self.displayStatsPerformed.append(dfResult, ignore_index = True)
		self.resultCollection.save_test(dfResult)
		self.plotter.save_statistics(self.saveStatsPerformed,self.displayStatsPerformed)
		self.reset_identification_process()		
		
	def plot_sigLines_and_pValue(self):
		'''
		'''
		
		ax = self.saveClickCoords['axis']
		
		testResult = self.saveClickCoords['testResult']
		
		yAxisMin, yAxisMax = ax.get_ylim()
		xPositionCLick1, xPositionCLick2 = self.saveClickCoords['indexClick1'], \
											self.saveClickCoords['indexClick2']
											
		yPositionClick1, yPositionClick2 = self.saveClickCoords['xyCoordsClick1'][1], \
											self.saveClickCoords['xyCoordsClick2'][1]
		
		yPositionMax = max([yPositionClick1, yPositionClick2])
		midXPosition = (xPositionCLick1 + xPositionCLick2)/2

		#numerically determined
		h = yPositionMax + abs(yAxisMax*0.75-yPositionMax)*0.43 
		
		
		xValuesLine = [xPositionCLick1,xPositionCLick1,
					   xPositionCLick2,xPositionCLick2] 
					   
		yValuesLine = [yPositionClick1,h,
					   h,yPositionClick2]
					   
		line = ax.plot(xValuesLine,yValuesLine,**signLineProps)
		pValue = return_readable_numbers(testResult[1]) 
		#pValue = '{:.2e}'.format(testResult[1])
		text_stat = ax.text(midXPosition, h, pValue, **standardTextProps)
		## saving text items and line
		self.saveClickCoords['text'] = text_stat
		self.saveClickCoords['line'] = line[0]
		## saving line props
		self.saveClickCoords['linesAndText'] = {'x':xValuesLine,'y':yValuesLine,
												'pVal':pValue,'textPos':(midXPosition,h)}
		self.saveClickCoords['axisIndex'] = self.plotter.get_number_of_axis(ax)
		
		if h > yAxisMax - 0.12* yAxisMax:
			for axis in self.plotter.get_axes_of_figure():
				axis.set_ylim([yAxisMin, yAxisMax+yAxisMax*0.20])
		
		self.save_test(self.saveClickCoords)
		
		self.plotter.redraw()

		
		
		
def compare_two_groups(testSettings,groupList):
		
		paired = testSettings['paired']#self.selectedTest['paired']
		test = testSettings['test']#self.selectedTest['test']
		mode = testSettings['mode']# self.selectedTest['mode']	
		#print(paired,test,mode)
		if test in ['t-test','Welch-test']:
			if test == 'Welch-test':
				equalVariance = False
			else:
				equalVariance = True
			if paired:
				try:
					if mode == 'two-sided [default]' or mode == 'greater':
						group1, group2 = groupList
					elif mode == 'less':
						group1, group2 = groupList[1], groupList[0]
					
					testResult = ttest_rel(group1,group2,nan_policy='omit')
					if mode != 'two-sided [default]':
						if mode == 'greater':
							mult = 1 
						else: 
							mult = -1
						if testResult[0] < 0:
							testResult = (mult*testResult[0], 1-testResult[1]/2)
						else:	
							testResult = (mult*testResult[0], testResult[1]/2)
							
				except ValueError:
					tk.messagebox.showinfo('Error ..','Can not handle unequal data size.')
					testResult = (np.nan, np.nan)
					
			else:
				try:
					if mode == 'two-sided [default]' or mode == 'greater':
						group1, group2 = groupList
					elif mode == 'less':
						group1, group2 = groupList[1], groupList[0]
					testResult = ttest_ind(group1,group2, nan_policy='omit', 
																equal_var = equalVariance)
					if mode != 'two-sided [default]':
						
						if mode == 'greater':
							mult = 1 
						else: 
							mult = -1
							
						if testResult[0] < 0:
							testResult = (mult*testResult[0], 1-testResult[1]/2)
						else:	
							testResult = (mult*testResult[0], testResult[1]/2)
				except:
					testResult = (np.nan,np.nan)
					
		elif test == 'Whitney-Mann U [unpaired non-para]':
			if mode == 'two-sided [default]':
				alt_test = 'two-sided'
			else:
				alt_test = mode
			try:
				testResult = mannwhitneyu(groupList[0],groupList[1],alternative=alt_test)
			except:
				testResult = (np.nan,np.nan)
			
		elif test == 'Wilcoxon [paired non-para]': 
		
			groupList = [group[~np.isnan(group)] for group in groupList]
			try:
				testResult = wilcoxon(groupList[0],groupList[1],correction=True)		
			
			except ValueError:
				
				quest = tk.messagebox.askquestion('Unequal N',
						'Wilcoxon cannot handle unequal N. Would you like to get an equal subset of the two groups?')			
				if quest == 'yes':
					min = [group.shape[0] for group  in groupList]
					groupList = [np.random.choice(group,min,replace=False) for group in groupList]
					testResult = wilcoxon(groupList[0],groupList[1])		
				else:
					testResult = (np.nan,np.nan)
					
			if mode != 'two-sided [default]':
			
						if mode == 'greater':
							mult = 1 
						else: 
							mult = -1
							
						if testResult[0] < 0:
							testResult = (mult*testResult[0], 1-testResult[1]/2)
						else:	
							testResult = (mult*testResult[0], testResult[1]/2)
		
			 
				
		return testResult

def compare_multiple_groups(test,data):
	'''
	'''
	
	if test == '1-W-ANOVA':
		if True:
			testResult= f_oneway(*data)
			#testResult = (stat,pValue)
		else:
			testResult = (np.nan,np.nan)
	return testResult
		
def estimateQValue(pv, m=None, verbose=False, lowmem=False, pi0=None):
    '''
    =============================================
    REFERENCE : https://github.com/nfusi/qvalue
    =============================================
    
    Estimates q-values from p-values
    Args
    =====
    m: number of tests. If not specified m = pv.size
    verbose: print verbose messages? (default False)
    lowmem: use memory-efficient in-place algorithm
    pi0: if None, it's estimated as suggested in Storey and Tibshirani, 2003.
         For most GWAS this is not necessary, since pi0 is extremely likely to be
         1
    '''
    assert(pv.min() >= 0 and pv.max() <= 1), "p-values should be between 0 and 1"

    original_shape = pv.shape
    pv = pv.ravel()  # flattens the array in place, more efficient than flatten()

    if m is None:
        m = float(len(pv))
    else:
        # the user has supplied an m
        m *= 1.0

    # if the number of hypotheses is small, just set pi0 to 1
    if len(pv) < 100 and pi0 is None:
        pi0 = 1.0
    elif pi0 is not None:
        pi0 = pi0
    else:
        # evaluate pi0 for different lambdas
        pi0 = []
        lam = np.arange(0, 0.90, 0.01)
        counts = np.array([(pv > i).sum() for i in np.arange(0, 0.9, 0.01)])
        for l in range(len(lam)):
            pi0.append(counts[l]/(m*(1-lam[l])))

        pi0 = np.array(pi0)

        # fit natural cubic spline
        tck = interpolate.splrep(lam, pi0, k=3)
        pi0 = interpolate.splev(lam[-1], tck)
        if verbose:
            print("qvalues pi0=%.3f, estimated proportion of null features " % pi0)

        if pi0 > 1:
            if verbose:
                print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
            pi0 = 1.0

    assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0

    if lowmem:
        # low memory version, only uses 1 pv and 1 qv matrices
        qv = np.zeros((len(pv),))
        last_pv = pv.argmax()
        qv[last_pv] = (pi0*pv[last_pv]*m)/float(m)
        pv[last_pv] = -np.inf
        prev_qv = last_pv
        for i in range(int(len(pv))-2, -1, -1):
            cur_max = pv.argmax()
            qv_i = (pi0*m*pv[cur_max]/float(i+1))
            pv[cur_max] = -np.inf
            qv_i1 = prev_qv
            qv[cur_max] = min(qv_i, qv_i1)
            prev_qv = qv[cur_max]

    else:
        p_ordered = np.argsort(pv)
        pv = pv[p_ordered]
        qv = pi0 * m/len(pv) * pv
        qv[-1] = min(qv[-1], 1.0)

        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(pi0*m*pv[i]/(i+1.0), qv[i+1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = np.zeros_like(qv)
        qv[p_ordered] = qv_temp

    # reshape qvalues
    qv = qv.reshape(original_shape)

    return qv,pi0
		
		
	  
           








