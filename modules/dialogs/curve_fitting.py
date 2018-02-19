import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts

import seaborn as sns

import numpy as np
import pandas as pd

from itertools import chain
from collections import OrderedDict

from scipy.optimize import curve_fit
from scipy import interpolate

from modules.utils import *
from modules.dialogs.import_subset_of_data import importDataFromDf
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame
from pandastable import Table, TableModel

import time

labelsTexts = ['Model: ','Degree [poly.]: ','Name of fit:','Enter x values:']


class _HelperCurveFitter(object):
	'''
	Add here your function that you would like to fit. 
	Step 1 - Define your function 
	Step 2 - Add A name and the function in the OrderedDict Object: self.curveFitFunctions
	Step 3 - Fit your data as it will be present in InstantClue from now on.
	'''
	def __init__(self):
		
		self.curveFitFunctions = OrderedDict([('linear fit',self.linear_fit),
						  			  ('A exp(b*x)', self.exponential_fit_1term),
						  			  ('A cos(x * freq + phase)+b',self.cosine_fit),
						  			  ('A (1 - exp(-k * x))',self.exponential_fit_one_e),
						  			  ('A exp(b*x) + C exp(d*x)',self.exponential_fit_2term),
						  			  ('Michaelis Menten (Vmax*x)/(Km+x)',self.michaelis_menten),
						  			  ('Gaussian fit A exp(-(x-mu)^2/(2*sigma^2))',self.gaussian_fit),
						  			  ('Weibull Dist. (a * b) x^(b-1)*exp(-a * x^b)',self.weibull_fit)])
	
	
	@property
	def get_fit_functions(self):
		return self.curveFitFunctions
		
		
	def linear_fit(self,x,m,b):
		'''
		'''
		return m*x+b
	
			
	def michaelis_menten(self,x,Vmax,Km):
		'''
		'''
		a = Vmax * x
		b = Km + x
		return (a) / (b)
	

	def exponential_fit_1term(self,x,a,b):
		'''
		'''
		return a*np.exp(b*x)
    	

	def cosine_fit(self,x,amplitude,phase,offset, omega = (2*np.pi)/23.6):
		'''
		'''
		return amplitude*np.cos(x*omega + phase)+offset
	
	
	def exponential_fit_2term(self,x,a,b,c,d):
		'''
		'''
		return a*np.exp(b*x) + c*np.exp(d*x)
    	

	def exponential_fit_non_e(self,x,a,b,c):
		'''
		'''
		return a*b**x+c
    
	def exponential_fit_one_e(self,x,A,k,y0):
		'''
		'''
		return A*(1-np.exp(-k*x)) + y0
    	
    	
	def gaussian_fit(self,x,A,mu,sigma):
		'''
		'''
		return A*np.exp(-(x-mu)**2/(2.*sigma**2))
		
	def weibull_fit(self,x,a,b):
		'''
		'''
		return a*b*x**(b-1)*np.exp(-a*x**b)


class curveFitCollection(object):
	'''
	'''
	def __init__(self):
		'''
		'''
		self.fitCollection = OrderedDict() 
	
	def save_performed_fit(self,fitIdName, columnNames,xValues, data, fittingFunc, dfId):
		'''
		'''
		self.fitCollection[fitIdName] = dict(columnNames = columnNames,
											 xValues = xValues,
											 fitData = data,
											 fittingFunc = fittingFunc,
											 dataFrameID = dfId)
	
	def remove_fits_by_dataId(self,dataId):
		'''
		Removes fit by data id (data collection). Happens for example
		when user deletes a complete data frame.
		'''
		toBeDeleted = []
		for fitId, props in self.fitCollection.items():
			if props['dataFrameID'] == dataId:
				toBeDeleted.append(fitId)
				
		for fitId in toBeDeleted:
			del self.fitCollection[fitId]
			
	def curve_fits_from_same_df(self,fitIds):
		'''
		'''	
		dfIds = [self.fitCollection[fitId]['dataFrameID'] for fitId in fitIds]	
		if any(dfId != dfIds[0] for dfId in dfIds):
			return False
		else:
			return True
				
	
	def get_columns_of_fitIds(self,fitIdList):
		'''
		'''
		collectList = []
		for fitId in fitIdList:
			columnNames = self.fitCollection[fitId]['columnNames']
			# filter uniques 
			columnsFilter = [col for col in columnNames if col not in collectList]
			collectList = collectList + columnsFilter
		
		return collectList		  

class curveFitter(object):
	
	def __init__(self, columns,dfClass, dataTreeview,curveFitCollection):
		helperFit = _HelperCurveFitter()
		
		helperFitFuncs = helperFit.get_fit_functions
		
		self.helperFitFuncs = helperFitFuncs
		self.fittingFunctions = ['polynomial fit','cubic spline'] + list(helperFitFuncs.keys())
		
		self.degreeVariable = tk.StringVar()
		self.fittingFunction = tk.StringVar()
		self.nameOfFit = tk.StringVar()
		self.calculateAUC = tk.BooleanVar(value = True)
		
		self.columnEntryDict = OrderedDict() 
		
		self.numericColumns = columns
		self.numbNumericColumns = len(columns)
	
		
		self.dfClass = dfClass
		self.dfID = self.dfClass.get_id_of_current_data()
		self.df = dfClass.get_current_data_by_column_list(self.numericColumns)
		self.dfLength = len(self.df.index)
		self.nanString = dfClass.replaceObjectNan
		
		self.dataTreeview = dataTreeview
		self.curveFitCollection = curveFitCollection # place to store fits
		
		self.build_toplevel() 
		self.build_widgets()		
		self.toplevel.wait_window() 	
	
	def close(self,event=None):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Curve fitting') 
		popup.bind('<Escape>',self.close)
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=370
		h=530
		self.toplevel = popup
		self.center_popup((w,h))
		
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	  		
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		self.cont.grid_columnconfigure(1,weight=1)
 		self.cont.grid_columnconfigure(0,weight=1,minsize=110)
 		self.cont.grid_rowconfigure(5,weight=1)
 		labelTile = tk.Label(self.cont, text = 'Fit each row to x-values by given model.',
 												**titleLabelProperties)
                            
 		labelTile.grid(padx=10, pady=15, columnspan=6, sticky=tk.W)
 		for n,text in enumerate(labelsTexts):
 			label = tk.Label(self.cont, text = text, bg=MAC_GREY)
 			label.grid(row=n+1,column=0,pady=2,padx=3,sticky=tk.W,columnspan=2)
 			
 		comboboxDegree = ttk.Combobox(self.cont,textvariable = self.degreeVariable, values = list(range(0,self.numbNumericColumns)))
 		comboboxDegree.insert(0,2)
 		
 		entryNameOfFit = tk.Entry(self.cont, textvariable = self.nameOfFit)
 		entryNameOfFit.configure(highlightbackground="#4C626F", 
 			highlightcolor="#4C626F",highlightthickness=2)
 		
 		self.nameOfFit.set('Curve fit {}'.format(len(self.curveFitCollection.fitCollection)))
 		
 		optionmenuFit =  ttk.OptionMenu(self.cont, self.fittingFunction,
 								'polynomial fit',*self.fittingFunctions)
 		
 		optionmenuFit.grid(row = 1, column = 1, pady = 3, padx = 5,sticky = tk.EW, columnspan = 2) 
 		comboboxDegree.grid(row = 2, column = 1, pady = 3, padx = 5,sticky = tk.EW, columnspan = 2)
 		entryNameOfFit.grid(row = 3, column = 1, pady = 3, padx = 5,sticky = tk.EW, columnspan = 2)
 		
 		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=5,sticky=tk.EW,pady=(2,8), columnspan=5,padx=3)
 		
 		columnFrame = tk.Frame(self.cont,bg=MAC_GREY,relief=tk.GROOVE,bd=2)
 		columnFrame.grid(row=5,column=0,sticky=tk.NSEW,columnspan=3,pady=5,padx=2)
 		vertFrame = VerticalScrolledFrame(columnFrame)
 		vertFrame.pack(expand=True,fill=tk.BOTH)
 		vertFrame.interior.grid_columnconfigure(0,weight=1,minsize=130)
 		vertFrame.interior.grid_columnconfigure(1,minsize=40)
		
 		for n,column in enumerate(self.numericColumns):
 		
 			labelColumn = tk.Label(vertFrame.interior, text = '{} :'.format(column), bg=MAC_GREY) 
 			labelColumn.grid(row = n,column = 0,padx = 4,pady = 2, sticky = tk.E)
 			entryXValue = ttk.Entry(vertFrame.interior) 
 			entryXValue.grid(row = n,column = 1,padx = 4, pady = 2, sticky = tk.W)
 			entryXValue.insert(tk.END,n)
 			self.columnEntryDict[column] = entryXValue
 		## import label	
 		#vertFrame.grid(row=6,column=0,columnspan=2,sticky=tk.E,padx=2,pady=5)
 		importLabel = tk.Label(self.cont, text = 'Import values from: ', bg = MAC_GREY)	
 		
 		## checkbutton if AUC should be calculated
 		checkbuttonAUC  = ttk.Checkbutton(self.cont, variable = self.calculateAUC, text = 'Calculate Area under Curve')
 		
 		
 		## define buttons
 		applyButton = ttk.Button(self.cont, text = 'Fit', command = self.fit_data,width=5)
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close,width=6)
 		#importFileButton = ttk.Button(self.cont, text = 'File')
 		importFromDataButton = ttk.Button(self.cont, text = 'Data', command = self.import_data,width=6)
 		
 		## taking n from above + 5 -> below entries ..
 		rowButtons = n+7
 		
 		importLabel.grid(row = rowButtons, column = 0, columnspan = 2, sticky = tk.W, padx = 5)
 		#importFileButton.grid(row = rowButtons, column = 1, sticky = tk.E, columnspan=2)
 		importFromDataButton.grid(row= rowButtons, column = 1, sticky = tk.E, columnspan=2,padx=5)
 		
 		checkbuttonAUC.grid(row = rowButtons+1, column = 0, sticky = tk.W,padx = 3,
 							pady = (10,2), columnspan = 2)
 		
 		applyButton.grid(row = rowButtons+2, column = 0, sticky = tk.W+tk.S,pady = 10,padx=5)
 		closeButton.grid(row = rowButtons+2, column = 2,sticky = tk.E+tk.S,pady = 10,padx=5)
 		#self.cont.grid_rowconfigure(rowButtons+2,weight=1)
 	 		
	
	def import_data(self):
		'''
		'''
		importer = importDataFromDf(self.dfClass,
 						title = 'Select data from preview as your x values.'+
 						' They must match either in row- or column-number the'+
 						' selected numeric columns: {}.'.format(self.numbNumericColumns),
 						requiredDataPoints = self.numbNumericColumns)
 		## get selected data				
		selectionData = importer.get_data_selection()
		## ensure that df Class has correct data selected (if user change it to select data) 
		#self.dfClass.set_current_data_by_id(self.dfID)
		
		if selectionData is None:
			return

		
		del importer
		self.insert_new_x_data(selectionData.values)
		
 						
	def insert_new_x_data(self,newXValues):

 		'''
 		'''
 		n = 0
 		for column,entry in self.columnEntryDict.items():
 			entry.delete(0,tk.END) 
 			entry.insert(0,str(newXValues.item(n)))
 			n += 1
 			
 						
	def get_x_values(self):
		'''
		'''
		try:
			xValues = [float(entry.get()) for entry in self.columnEntryDict.values()]
			
			return np.array(xValues)
		except:
			tk.messagebox.showinfo('Error ..',
									'There was an error converting your x-value entries to floats.',
									parent = self.toplevel)
			return 
		
	def get_column_names(self,fittingFunc):
		'''
		'''
		#columnNames  = get_elements_from_list_as_string(self.numericColumns).replace(' ','')
		fitName = self.nameOfFit.get() 
		
		if fittingFunc in self.helperFitFuncs:
			
			labelColumns = ['Coeff_{}'.format(fitName ),
						   'StdDevCoeff_{}'.format(fitName ),
						   'R^2_{}'.format(fitName )]

		elif fittingFunc == 'polynomial fit':
			
			labelColumns = ['Coeff_{}'.format(fitName ),
							'R^2_{}'.format(fitName )]

		elif fittingFunc == 'cubic spline':
		
			labelColumns = ['quadCubicSpline_{}'.format(fitName),
							'SumSquaRes_{}'.format(fitName)]
		   
		if self.calculateAUC.get():
			labelColumns.append('AUC_{}'.format(fitName))
			
		labelColumnsEval = 	[self.dfClass.evaluate_column_name(name) for name in labelColumns]
		
		return labelColumnsEval
		
	def fit_data(self,castProgressbar = True):
		'''
		'''
		if self.nameOfFit.get() in self.curveFitCollection.fitCollection:
			tk.messagebox.showinfo('Name exists ..','Name of curve fit has been used. Please rename.')
			return
		
		self.xValues = self.get_x_values()
		
		if self.xValues is None:
			return
		
		fittingFunc = self.fittingFunction.get()
		
		calcAUC = self.calculateAUC.get()
		if calcAUC:	
			# get 4 times as many values
			xLinAUC = self.calculate_auc_x_values(self.xValues) 
		else:
			xLinAUC = None
		
		
		if castProgressbar:
			## object reportProgress is used to prohibit update of the progress bar at every index
			## which would consume too much time. Instead we report every 10 % by using
			## if index.int % self.reportProgress == 0.
			self.reportProgress = int(float(self.dfLength * 0.1))
			self.progressbar = Progressbar(title= 'Curve fitting')
			self.progressbar.update_progressbar_and_label(2,'Got x values')
			
		if fittingFunc in self.helperFitFuncs:
		
			data = (self.df.apply(lambda row: self.curve_fit(self.xValues,row,fittingFunc,
									calcAUC = calcAUC, xLinAUC = xLinAUC), axis=1)).apply(pd.Series)

			
		elif fittingFunc == 'polynomial fit':
		
			## check if input for degree makes sense
			degree = self.get_degree_of_poly()
			if degree is None:
				self.progressbar.close()
				return
			
			data = (self.df.apply(lambda row: self.fit_polynomial(self.xValues,row,degree,
									calcAUC = calcAUC, xLinAUC = xLinAUC), axis=1)).apply(pd.Series)
			

	
			
		elif fittingFunc == 'cubic spline':
		
			s = ts.askfloat('Value for s...', prompt = 'Please provide a value for s.'+
								'\nControlling the trade-off between closeness (s=0) and'+
								' smoothness of fit.\nLarger s means more smoothing while'+
								' smaller values of s indicate less smoothing.',
							 	initialvalue = 0, minvalue = 0, parent=self.progressbar.toplevel) 
			if s is None:
				return 
				
			data = (self.df.apply(lambda row: self.fit_spline(self.xValues,row,s,
								calcAUC = calcAUC, xLinAUC = xLinAUC), axis=1)).apply(pd.Series)
		
		columnNames = self.get_column_names(fittingFunc)
		data.columns = columnNames
		self.add_fit_data_to_source_df(data)
		dataTypeList = self.dfClass.get_data_types_for_list_of_columns(columnNames)
		self.dataTreeview.add_list_of_columns_to_treeview(self.dfID,dataTypeList,columnNames)
		## report being done..
		if castProgressbar:
			self.progressbar.update_progressbar_and_label(99,'Saved fit data and close ..')
			self.progressbar.close()
		
		self.save_fit(data)	
		
	 		
	def fit_polynomial(self,x,yRaw, degree, updatePB = True, calcAUC = True, xLinAUC = None):
		'''
		Actually fits the polynominal using numpy's polyfit function.
		Calculates also R^2 and optional the AUC.
		'''
		x,y = self.filter_x_y_for_nan(x,yRaw)
		
		if x.size < 3:
			if calcAUC:
				return self.nanString,np.nan,np.nan
			else:
				return self.nanString,np.nan
		try:
			polynominalFit =  np.polyfit(x,y, deg=degree)
			polynominal = np.poly1d(polynominalFit)
			coeffString = get_elements_from_list_as_string(polynominal)
			
			yFit = polynominal(x) 
			rSq = self.calculate_r_squared(x,y,yFit)
			rowIntIdx = self.df.index.get_loc(yRaw.name)
			if updatePB and rowIntIdx % self.reportProgress == 0 and self.dfLength != 0: 
				## updateText does not change label text - stays the same anyway
				self.progressbar.update_progressbar_and_label(rowIntIdx/self.dfLength*100,
											'Calculating ..', 
											updateText = True)
		except:
		
			if calcAUC:
				return self.nanString,np.nan,np.nan
			else:
				return self.nanString,np.nan
		
		
		if calcAUC:
			
			areaUnderCurve = np.trapz(y = polynominal(xLinAUC),
									  x = xLinAUC)
									  
			return coeffString,rSq,areaUnderCurve
		else:	
		
			return coeffString,rSq 
		
		
		
	def fit_spline(self,x,yRaw,s, updatePB = True, calcAUC = True, xLinAUC = None):
		'''
		fitting a cubic spline
		'''
		try:
			x,y = self.filter_x_y_for_nan(x,yRaw)
			tck = interpolate.splrep(x,y, s = s, full_output = 1)
		except:
			if calcAUC:
				return 'Error', np.nan, np.nan
			else:
				return 'Error', np.nan
				
		tupleVector = tck[0]

		formatString  = '{};{};{}'.format(get_elements_from_list_as_string(tupleVector[0]),
							 get_elements_from_list_as_string(tupleVector[1]),
							 tupleVector[2])
		RSQ = tck[-3]
		rowIntIdx = self.df.index.get_loc(yRaw.name)
		if updatePB and rowIntIdx % self.reportProgress == 0: 
			self.progressbar.update_progressbar_and_label(rowIntIdx/self.dfLength*100,
											'Calculating ..', 
											updateText = True)
		if calcAUC:
			areaUnderCurve = np.trapz(interpolate.splev(xLinAUC,tck[0],der=0),
									  x = xLinAUC)
									  
			return formatString , RSQ, areaUnderCurve
			
		else:
			return formatString , RSQ
			
			
			
			
			
	def curve_fit(self,x,yRaw,fittingFunc,updatePB = True, calcAUC = True, xLinAUC = None):
		'''
		Using scipy's curve_fit function to fit any data.
		'''
		try:
			x,y = self.filter_x_y_for_nan(x,yRaw)
			# popt _ optimal parameter, pcov - covariance
			popt, pcov  = curve_fit(self.helperFitFuncs[fittingFunc],xdata = x, ydata = y)
			yFit =  self.helperFitFuncs[fittingFunc](x, *popt)
			
			##  standard deviation errors on the parameters: perr
			perr = np.sqrt(np.diag(pcov))
			rSquared = self.calculate_r_squared(x,y,yFit)
			
			poptString = get_elements_from_list_as_string(popt)
			pErrorString = get_elements_from_list_as_string(perr)
			
			rowIntIdx = self.df.index.get_loc(yRaw.name)
			if updatePB and rowIntIdx % self.reportProgress == 0: 
				self.progressbar.update_progressbar_and_label(rowIntIdx/self.dfLength*100,
											'Calculating ..', 
											updateText = True)
			if calcAUC:
			
				
				areaUnderCurve = np.trapz(y = self.helperFitFuncs[fittingFunc](xLinAUC, *popt),
									  x = xLinAUC)
				return poptString,pErrorString,rSquared, areaUnderCurve
									  
			else:
			
				return poptString,pErrorString,rSquared
		
		except:
			if calcAUC:
			
				return self.nanString,self.nanString,np.nan, np.nan
				
			else:
				return self.nanString,self.nanString,np.nan
				
				
	def calculate_auc_x_values(self,x,multiplyNBy = 8):
		'''
		'''
		x0 = x.min() 
		x1 = x.max()
		numPoints = x.size * multiplyNBy
		xLinspace = np.linspace(x0, x1, num = numPoints, endpoint = True)
		return xLinspace 

   
	def filter_x_y_for_nan(self,x,y):
		'''
		x must be a numpy array
		'''
		y = np.array(y)
		
		x = x[~np.isnan(y)]
		y = y[~np.isnan(y)]
		
		order =  np.argsort(x)
		
		return x[order],y[order]

			
	def calculate_r_squared(self,xRaw,yRaw,yFit):
		'''
		Calculate R^2 
		'''
		
		ybar = np.sum(yRaw)/len(yRaw) 
		
		ssreg = np.sum((yFit-ybar)**2) 
		sstot = np.sum((yRaw - ybar)**2)
		
		rSq = ssreg/sstot
		
		return rSq
		
	def get_degree_of_poly(self):
		'''
		'''
			
		try:
			degree = int(float(self.degreeVariable.get()))
			if degree == self.numbNumericColumns:
				tk.messagebox.showinfo('Error ..',
										'Degree of polynomial is equal the numbers of numeric columns.'+
										'This would result always in a perfect fit. Aborting ..',
										parent = self.toplevel)
				return
		except:
			tk.messagebox.showinfo('Error ..',
								   'Could not convert degree input to number.',
								   parent=self.toplevel)
			return
			
		return degree	
							
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y))) 		
         	

	def add_fit_data_to_source_df(self,df):
		'''
		add collected data frame with fitting data to currently selected data frame
		'''
		self.dfClass.join_df_to_currently_selected_df(df)


	def save_fit(self,data):
		'''
		Saving the performed fit
		'''
		self.curveFitCollection.save_performed_fit(fitIdName = self.nameOfFit.get(),columnNames = self.numericColumns,
											  xValues = self.xValues,data = data,fittingFunc = self.fittingFunction.get(),
											  dfId = self.dfID)
	
         	
class customChartLayout(object):
	
	def __init__(self, dfClass, gridLayout = (4,4), colorScheme = 'Blues'):
	
		self.row, self.column = gridLayout
		self.colorScheme = colorScheme
		
		self.frameGridDict = OrderedDict()
		self.indexFrameDict = OrderedDict()
		self.subplotNumDataIdx = dict()
		
		self.df = dfClass.get_current_data()
		
		self.maxRowsPerSubplot = tk.StringVar()
		
		self.onMotionLabel = None
		
		self.build_toplevel() 
		self.build_widgets()
		
		
		self.toplevel.wait_window() 	
	
	def close(self):
		'''
		Close toplevel
		'''
		self.toplevel.destroy() 

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Grid layout') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=1100
		h=565
		self.toplevel = popup
		self.center_popup((w,h))
		
	def build_widgets(self):
 		'''
 		Builds the dialog for interaction with the user.
 		'''	 
 		
 		
 		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
 		self.cont.pack(expand =True, fill = tk.BOTH)
 		
 		#self.cont.grid_columnconfigure(0,weight=1 ,minsize=250)
 		self.cont.grid_columnconfigure(8,weight=1, minsize=150)
 		self.cont.grid_rowconfigure(5,weight=1)
 		
 		labelTile = tk.Label(self.cont, text = 'Define layout for displaying curve fitting results.'+
 											   '\nDrag & Drop desired data rows onto the grid layout.'+
 											   '\nYou can combine multiple rows to plot them in one subplot.'+
 											   ' Shown numbers represent the index of selected rows.',
 												**titleLabelProperties)
                            
 		labelTile.grid(padx=10, pady=15, columnspan=8, sticky=tk.W)
 		
 		
 		clearAllButton = ttk.Button(self.cont, text = 'Clear all', command = self.clear_all_frames) 
 		fillDownButton = ttk.Button(self.cont, text = 'Fill down', command = self.fill_down)
 		labelMaxNumber = tk.Label(self.cont, text = 'Max rows per subplot: ', bg = MAC_GREY)
 		
 		entryMaxNumber = ttk.Entry(self.cont, textvariable = self.maxRowsPerSubplot, width=4)
 		self.maxRowsPerSubplot.trace(mode='w',callback = self.generate_colors)
 		self.maxRowsPerSubplot.set(4)
 		
 		clearAllButton.grid(row = 1, column = 5, sticky=tk.W)
 		fillDownButton.grid(row = 1, column = 4, sticky=tk.W) 
 		labelMaxNumber.grid(row = 1, column = 2, sticky=tk.W)
 		entryMaxNumber.grid(row = 1, column = 3, sticky=tk.W)
 		 		
 		
 		self.cont_preview = tk.Frame(self.cont, background = MAC_GREY)
 		self.cont_gridlayout = tk.Frame(self.cont)
 		
 		self.cont_preview.grid(row=5,column=0,sticky = tk.NSEW)
 		self.cont_gridlayout.grid(row=5,column=1,sticky = tk.NSEW,columnspan=10) 
 		self.display_grid_layout()
 		
 		applyButton = ttk.Button(self.cont, text = 'Done', command = self.extract_index_per_subplot)
 		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
 		
 		applyButton.grid()
 		closeButton.grid()
 		
 		self.show_data()
 		
 		
	def display_grid_layout(self):
		'''
		'''
		self.framesCreated = []
		for gridRow in range(self.row):
			for gridColumn in range(self.column):
				frame = tk.Frame(self.cont_gridlayout, bd = 2, relief = tk.GROOVE)
				frame.grid(row = gridRow,column = gridColumn,sticky = tk.NSEW)
				
				self.cont_gridlayout.grid_rowconfigure(gridRow,weight=1, minsize=60)
				self.cont_gridlayout.grid_columnconfigure(gridColumn,weight=1, minsize=60)
				
				
				gridKey = '{}_{}'.format(gridRow,gridColumn)
				self.frameGridDict[gridKey] = frame
				self.add_bindings_to_frame(frame)
				self.framesCreated.append(frame)
				self.indexFrameDict[frame] = []

	def show_data(self):
		'''
		'''
		self.pt = Table(self.cont_preview,
						dataframe = self.df,
						showtoolbar=False,
						showstatusbar=False) 
		self.pt.show()
		
		# overwrite B1-Motion standard of pandastable
		self.pt.bind('<B1-Motion>', self.on_motion)
		self.pt.rowheader.bind('<B1-Motion>', self.on_motion)
		# end drag & drop
		self.pt.bind('<ButtonRelease-1>', self.on_release)
		self.pt.rowheader.bind('<ButtonRelease-1>', self.on_release)	
		
			 
	def on_motion(self,event):
		'''
		'''
		
		x = self.toplevel.winfo_pointerx() - self.toplevel.winfo_rootx()
		y = self.toplevel.winfo_pointery() - self.toplevel.winfo_rooty()
		
		if self.onMotionLabel is None:
		
			self.indexSelection = [self.pt.model.df.index.tolist()[row] for row in self.pt.multiplerowlist]
			columnSel = self.pt.multiplecollist[0]
			self.contentSelection = [self.pt.model.getValueAt(row,columnSel) for row in self.pt.multiplerowlist]
			self.onMotionLabel = tk.Label(self.cont,text=str(self.contentSelection))
			self.onMotionLabel.place(x= x-8,y=y-8, anchor= tk.NE)
					
		else:
			self.onMotionLabel.place(x= x-8,y=y-8,anchor= tk.NE)
  
		self.grid_widget = self.cont.winfo_containing(event.x_root, event.y_root)  
		
		for frame in self.framesCreated:
			
			if frame == self.grid_widget:
			
				self.onMotionLabel.configure(fg='#4C626F')
				self.grid_widget.configure(bg=MAC_GREY)
				
			else:		
				frame.configure(bg='white')
				self.onMotionLabel.configure(fg='black')
	

	def on_release(self,event):
		'''
		'''
		if self.onMotionLabel is not None:
			self.onMotionLabel.destroy() 
			self.onMotionLabel = None

			coords = self.identify_frame(self.grid_widget)
			if coords is None:
				return
				
			self.add_drop_item()
			
		else: 
			pass 
				
					
	def clear_all_frames(self):
		'''
		Remove all widgets from created frames
		'''
		self.indexFrameDict.clear()
		
		for frame in self.frameGridDict.values():
			for widget in frame.winfo_children():
				widget.destroy()
			self.indexFrameDict[frame] = []
				
				
				
	def identify_frame(self,widget):
		'''
		'''
		for coords, frame in self.frameGridDict.items():
			if frame == widget:
				return coords
			else:
				pass
				
	def add_drop_item(self, indexList = None, contentSelection = None, frame = None):
		'''
		Adding label to the frame.
		'''
		if indexList is None and contentSelection is None:
			indexList = self.indexSelection
			contentSelection = self.contentSelection
			frame = self.grid_widget
		## resetting stored values	
		self.indexFrameDict[frame] = []
		nColumn = 0
		rowWidg = 0
		nColor = 0
		frame.grid_rowconfigure(0,weight=1)
		for index,label in zip(indexList,contentSelection):
		
			labelAdded = tk.Label(frame,
								 text=index,
								 bg=self.colors[nColor ])
			labelAdded.bind(right_click, self.delete_label)
			self.save_idx_and_frame(frame,index,labelAdded)
			
			
			CreateToolTip(labelAdded, text= label, pad = (1,1,1,1))
			
			if nColumn % 5 == 0:
				rowWidg += 1 
				nColumn = 0 
			#labelAdded.pack(anchor=tk.SW, side = sideW, fill='x', expand = True, padx=0.5)
			labelAdded.grid(row=rowWidg, column = nColumn, sticky=tk.EW,padx=0.5)			
			nColumn += 1
			nColor  += 1 
		
	def save_idx_and_frame(self,frame,indexLabel,labelWidget):
		'''
		'''
		
		self.indexFrameDict[frame].append((indexLabel,labelWidget))
		
		
	def fill_down(self):
		'''
		'''
		self.clear_all_frames()
		indexList = self.pt.model.df.index.tolist()
		lenDf = len(indexList)
		
		divIndex = [indexList[x:x+self.numColors] for x in range(0,lenDf,self.numColors)]
		neededFrames = self.framesCreated[:len(divIndex)]
		for frame, indexList in zip(neededFrames,divIndex):
			self.add_drop_item(indexList,indexList,frame)
		
		
	def delete_label(self,event):
		'''
		Delete the selected label
		'''
		
		w = event.widget
	
		for frame, indexAndLabelList in self.indexFrameDict.items():
			flattenList = list(sum(indexAndLabelList, ()))
			if w in flattenList:
				key = frame
				break
	
		listOfLabels = self.indexFrameDict[frame]
		listOfLabelsNew = []
		nColor = 0
		for n,indexAndLabel in enumerate(listOfLabels):
		
			label, addedLabel = indexAndLabel
			if addedLabel == w:
				keyToDel = n
			else:
				color = self.colors[nColor]	
				addedLabel.configure(bg=color)
				nColor += 1
				listOfLabelsNew.append((label,addedLabel))
		w.destroy()
		
		self.indexFrameDict[frame] = listOfLabelsNew
	
			
			
	def add_bindings_to_frame(self,frame):
		'''
		Add bindings for frames to change background on mouse enter
		'''
		def on_enter(event):
			w = event.widget
			w.configure(bg=MAC_GREY)
		
		def on_leave(event):
			w = event.widget
			w.configure(bg='white')
				
		frame.bind('<Enter>', on_enter)
		frame.bind('<Leave>', on_leave)
		
	def generate_colors(self,varname = None, elementname = None, mode=None):
		'''
		
		'''
		try:
			self.numColors = int(float(self.maxRowsPerSubplot.get()))
		except ValueError:
			return
		colors = sns.color_palette(self.colorScheme,self.numColors,desat=0.75)
		self.colors = [col_c(color) for color in colors]
		
	def extract_index_per_subplot(self):
		'''
		'''
		self.subplotNumDataIdx.clear()
		
		for n,indexAndLabelList in enumerate(list(self.indexFrameDict.values())):
			collectIdx = []
			for indexAndLabel in indexAndLabelList:
				index = indexAndLabel[0]
				collectIdx.append(index)
			self.subplotNumDataIdx[n] = collectIdx
			
		self.close()	
											
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))  			
 				
 				
 		
 		
 		
 		
 		
class displayCurveFitting(object):
	
	def __init__(self,dfClass,Plotter,courveFitCollection):
	
		self.rowNumGrid = tk.StringVar() 
		self.columnNumGrid = tk.StringVar()
		self.labelColumn = tk.StringVar()
		self.filterColumnsForFits = tk.BooleanVar(value=True)
		self.tightLayout = tk.BooleanVar(value=True)
		self.equalYLims = tk.BooleanVar(value=True)
		self.dfClass = dfClass
		self.columns = dfClass.get_columns_of_current_data()
		#self.data = dfClass.get_current_data()
		
		self.plotter = Plotter
		
		helperFit = _HelperCurveFitter()
		
		self.fittingFunctions = helperFit.get_fit_functions
		self.curveFitCollection = courveFitCollection
		
		self.customGridLayout = OrderedDict()
		self.subplotDataIdxDict = OrderedDict()
		
		self.curveFitsSelected = []
		self.find_curve_fittings()
		
		self.build_toplevel() 
		self.build_widgets()
		
		
		self.toplevel.wait_window() 	
	
	def close(self, reset = False):
		'''
		Close toplevel
		'''
		if reset:
			self.curveFitsSelected = [] 
			
		self.toplevel.destroy() 

	def build_toplevel(self):
	
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Display curve fits') 
         
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w=570
		h=565
		self.toplevel = popup
		self.center_popup((w,h))

	def build_widgets(self):
		'''
		Builds up widgets.
		'''
		self.cont= tk.Frame(self.toplevel, background =MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		
		self.cont.grid_columnconfigure(4,weight=1)
		self.cont.grid_rowconfigure(5,weight=1)
		
		labelTile = tk.Label(self.cont, text = 'Choose a grid layout and the curve fit'+
 							 ' you would like to display.\nIf you choose multiple fits, all will be displayed.'+
 							 '\nUse the custom-layout tool to define the order and combination of rows.',
 							 **titleLabelProperties)
		
		labelGridLayout = tk.Label(self.cont, text = 'Grid layout (rows,columns): ', bg=MAC_GREY)
		labelSubplotName = tk.Label(self.cont, text = 'Subplot annotation: ', bg = MAC_GREY)
		
		comboboxName = ttk.Combobox(self.cont, textvariable = self.labelColumn, 
											  values = self.columns,
											  width = 18)
		
		
		comboboxRow = ttk.Combobox(self.cont, textvariable = self.rowNumGrid, 
											  values = list(range(1,31)),
											  width = 8)
		comboboxColumn = ttk.Combobox(self.cont,  textvariable = self.columnNumGrid, 
											  values = list(range(1,31)),
											  width = 8)
		self.rowNumGrid.set('3')
		self.columnNumGrid.set('3')
		self.labelColumn.set('Row Number')
											  
		labelChooseFit  = tk.Label(self.cont, text = 'Choose curve fit to display: ', bg=MAC_GREY)
		scrollListBoxVer = ttk.Scrollbar(self.cont, orient = tk.VERTICAL)
		scrollListBoxHor = ttk.Scrollbar(self.cont, orient = tk.HORIZONTAL)
		self.listboxCurveFits = tk.Listbox(self.cont,xscrollcommand = scrollListBoxHor.set,
									yscrollcommand = scrollListBoxVer.set, selectmode = tk.MULTIPLE)

		scrollListBoxVer.config(command=self.listboxCurveFits.yview)
		scrollListBoxHor.config(command=self.listboxCurveFits.xview) 	
							
		#checkbuttonColumnShow = ttk.Checkbutton(self.cont, text = 'Show curve fit(s) only',
		#										variable = self.filterColumnsForFits,
		#										command = lambda listbox = self.listboxCurveFits: self.fill_listbox(listbox=listbox))	
												
		checkbuttonSameYLimit = ttk.Checkbutton(self.cont, text = 'Adjust y-limits', variable = self.equalYLims)
		
		CreateToolTip(checkbuttonSameYLimit,title_ = 'Adjust y-limits', 
					  text= 'If checked, y limits of all subplots will be the same.',
					  pad = (1,1,1,1))
					  
		checkbuttonTightLayout = ttk.Checkbutton(self.cont, text = 'Tight layout', variable = self.tightLayout)
		CreateToolTip(checkbuttonTightLayout,title_ = 'Tight layout', 
					  text= 'If checked, the space between subplots will be removed',
					  pad = (1,1,1,1))
					  
					  														
		plotButton = ttk.Button(self.cont, text = 'Plot', command =  self.set_chart_settings,width=6)                            
		closeButton = ttk.Button(self.cont, text = 'Close', command = lambda: self.close(reset=True),width=6)
		customLayoutButton = ttk.Button(self.cont, text = 'Customize Plot Order', command =  self.get_custom_grid_layout)  
		

		labelTile.grid(padx=10, pady=15, columnspan=8, sticky=tk.W) 
		                           
		labelGridLayout.grid(row=1,column=0, sticky=tk.W,padx=4,pady=2)
		comboboxRow.grid(row=1,column=1,padx=4,pady=2)
		comboboxColumn.grid(row=1,column=2,padx=4,pady=2)
		labelSubplotName.grid(row=2,column=0,padx=4,pady=2,sticky=tk.E)
		comboboxName.grid(row=2,column=1,padx=4,columnspan=2,sticky=tk.EW)
		
		labelChooseFit.grid(row=3,column=0, padx=3,pady=2)
		self.listboxCurveFits.grid(row=5, columnspan = 5,sticky=tk.NSEW,padx=3,pady=(3,0))
		scrollListBoxHor.grid(sticky=tk.EW,columnspan=4)
		scrollListBoxVer.grid(sticky=tk.NS+tk.W, row=5,column = 5)
		
		checkbuttonSameYLimit.grid(row=7,column = 0,columnspan=3,padx=3,pady=2,sticky=tk.W)
		checkbuttonTightLayout.grid(row=7,column = 2,columnspan=3,padx=(3,10),pady=2,sticky=tk.E)
		plotButton.grid(row = 8, column = 0, pady=4, padx=3, sticky=tk.W)
		closeButton.grid(row = 8, column = 3,pady=4, padx=10,sticky=tk.E)
		customLayoutButton.grid(row = 8, column = 1,pady=4, padx=5)
		self.fill_listbox(listbox = self.listboxCurveFits)
		
	def get_custom_grid_layout(self):
		'''
		'''
		customChartDialog = customChartLayout(self.dfClass, colorScheme ='Reds')	
		self.subplotDataIdxDict = customChartDialog.subplotNumDataIdx
		
	def find_curve_fittings(self):
		'''
		'''
		
		self.curveFits = list(self.curveFitCollection.fitCollection.keys())
 		
 		
	def fill_listbox(self, itemsToAdd = None, listbox = None):
		'''
		'''
		listbox.delete(0,tk.END)
		
		if itemsToAdd is None:
			if len(self.curveFits) != 0:
				itemsToAdd = self.curveFits
			else:
				return
		
		for item in itemsToAdd:
		
			listbox.insert(tk.END,item)
		
		self.columnsInListbox = itemsToAdd
		
	@property
	def curve_fits_to_plot(self):
		'''
		'''
		return self.curveFitsSelected
				
	def get_selected_curve_fit(self):
		'''
		'''
		self.curveFitsSelected = [self.columnsInListbox[idx] for idx in self.listboxCurveFits.curselection()]
		allFitsFromSameData =  self.curveFitCollection.curve_fits_from_same_df(self.curveFitsSelected)
		return allFitsFromSameData
						
	def set_chart_settings(self):
		'''
		Check fit settings.
		'''
		gridLayout = [int(float(value)) for value in [self.rowNumGrid.get(),self.columnNumGrid.get()]]
		adjustYlimit = self.equalYLims.get()
		tightLayout = self.tightLayout.get()
		labelColumn = self.labelColumn.get()
		#checks if same dataframe id underlying
		if self.get_selected_curve_fit():
			pass
		else:
			tk.messagebox.showinfo('Error ..',
					'Data fits can only be combined if the data are from same file.',
					parent=self.toplevel)
			return
		
		if len(self.curveFitsSelected) == 0:
		
			tk.messagebox.showinfo('Error..',
					'Select a curve fit to plot.',
					parent=self.toplevel)
			return
		self.plotter.set_curveFitDisplay_settings(gridLayout,
												  self.curveFitCollection.fitCollection,
												  self.subplotDataIdxDict,
												  adjustYlimit,
												  tightLayout,
												  labelColumn)
		self.close()
											
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))  			
 	
 	
 	
 
	
		
		
		
		
	
	
		

	