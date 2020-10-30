

#from modules.utils import *

class stackedAreaPlot(object):
	
	def __init__(self, numericColumns, categoricalColumns, dfClass, plotter, colorMap = None):
	
		self.axisDict = OrderedDict()
		self.categoricalColorDefinedByUser = dict()
		self.nanScatterColor = GREY
		self.dfClass = dfClass
		self.plotter = plotter
		self.dataID = self.dfClass.get_id_of_current_data()
		self.colorMap = colorMap
		
		self.numericColumns = numericColumns
		self.categoricalColumns = categoricalColumns
		
		print('here#')
		
		self.get_data()
		self.get_color()
		
		self.create_axis()
		self.fill_axis()

	def create_axis(self):
		'''
		'''
		if len(self.categoricalColumns) == 1:
			self.axisDict[0] = self.plotter.figure.add_subplot(111)		
		
	def get_data(self):
		'''
		'''
		
		self.data = self.dfClass.get_current_data_by_column_list(self.numericColumns+self.categoricalColumns)
		
		dfSum = self.data[self.numericColumns].sum()
		plotData = self.data.groupby(by=self.categoricalColumns)[self.numericColumns].sum().div(dfSum).cumsum()
		self.plotData = plotData.sort_values(self.numericColumns[0])
		print(self.plotData)
		
	def get_color(self):
		'''
		'''
		self.colorMapDict,layerMapDict, self.rawColorMapDict = get_color_category_dict(self.dfClass,
												self.categoricalColumns,
												self.colorMap, self.categoricalColorDefinedByUser,
												self.nanScatterColor)		
	def get_current_colorMapDict(self):
		'''
		'''
		if hasattr(self,'colorMapDict'):
			return self.colorMapDict
		else:
			return {}		

	def get_raw_colorMapDict(self):
		'''
		'''
		return self.rawColorMapDict
		
	def set_user_def_colors(self,categoricalColorDefinedByUser):
		'''
		'''
		self.categoricalColorDefinedByUser = categoricalColorDefinedByUser


	def update_colorMap(self,newCmap = None):
		'''
		'''
		if newCmap is not None:
			self.colorMap = newCmap
	
	def fill_axis(self):
		'''
		'''
		if len(self.categoricalColumns) == 1:
			
			print(self.colorMapDict)
			print(self.plotData)
			idx = range(len(self.numericColumns))
			dfIdx = self.plotData.index.tolist()
			for rowIdx in range(len(self.plotData.index)):
				yValues = self.plotData.iloc[rowIdx].values
				#self.axisDict[0].plot(idx,yValues)
				if rowIdx == 0:
					self.axisDict[0].fill_between(idx,[0]*len(self.numericColumns),yValues, 
						facecolor = self.colorMapDict[dfIdx[rowIdx]])
				else:
					self.axisDict[0].fill_between(idx,self.plotData.iloc[rowIdx-1].values,yValues,
						facecolor = self.colorMapDict[dfIdx[rowIdx]])
		
		
		
		
	
	