
from matplotlib import collections  as mc
from modules.utils import *

class clusterPlotting(object):
	
	def __init__(self, numericalColumns, plotter, dfClass, colorMap, scatterKwargs):
		'''
		'''
		self.numericalColumns = numericalColumns
		self.plotter = plotter
		self.dfClass = dfClass
		self.categoricalColorDefinedByUser = dict()
		self.sizeStatsAndColorChanges = dict()
		self.colorMapDict = None
		self.colorMap = colorMap
		self.dataID = self.plotter.get_dataID_used_for_last_chart()
		self.nanScatterColor = scatterKwargs['color']
		self.axisDict = dict()
		self.get_data()
		self.create_axis()
		self.fill_axis()
		
	def create_axis(self):
		'''
		'''
		rows = int(self.clustLabels.size / 3 + 1)
		for n in range(self.clustLabels.size):
			self.axisDict[n] = self.plotter.figure.add_subplot(rows, 3, n+1)
		self.plotter.figure.subplots_adjust(wspace=0.05, hspace = 0.05, right = 0.85)
		
	def fill_axis(self, specificAxis = None, subplotId = None):
		'''
		'''
		groupedData = self.data.groupby(self.clustColumn,sort=True)
		groupSizeData = groupedData.size()	
		
		scores = self.plotter.clusterEvalScores[len(self.plotter.clusterEvalScores)]	

		for n,(name, groupData) in enumerate(groupedData):
			if subplotId is not None:
				if n != subplotId:
					continue
			ax = self.axisDict[n] if specificAxis is None else specificAxis
			lineSegs = []
			if n == 1:
				ax.set_title(label='Silhouette : {}\nCalinski : {}'.format(round(scores['Silhouette'],2),
																	round(scores['Calinski'],0)))
			for id in groupData.index:
				yValues = groupData.loc[id][self.numericalColumns].values
				lineSegs.append(([(n,y) for n,y in enumerate(yValues)]))
			if 'color' not in self.data.columns:
				colors = 'darkgrey'
			else:
				colors = groupData['color']
			lc = mc.LineCollection(lineSegs, linewidths=1, colors = colors)
			ax.add_collection(lc)
			self.plotter.add_annotationLabel_to_plot(ax=ax, position = 'topright',
										text = 'n = {}'.format(groupSizeData.loc[name]))
			self.style_axis(ax, groupedData, name, specificAxis)
					

		#print(self.plotter.clusterEvalScores)
		#	calinski.append(scoreDict['Calinski'])
		#	silhouette.append(scoreDict['Silhouette'])
			
	
	def add_color_and_size_changes_to_dict(self,changeDescription,keywords):
		'''
		Adds information on how to modify the chart further
		'''
		self.sizeStatsAndColorChanges[changeDescription] = keywords
			
	
	def change_color_by_categorical_columns(self, categoricalColumns, specificAxis = None, subplotId = None):
		'''
		'''
		
		self.colorMapDict,layerMapDict, self.rawColorMapDict = get_color_category_dict(self.dfClass,
												categoricalColumns,
												self.colorMap, self.categoricalColorDefinedByUser,
												)		
		## update data if missing columns and add column 'color'
		self.data  = self.plotter.attach_color_data(categoricalColumns, self.data, 
													self.dataID, self.colorMapDict)											
		## group data 
		groupedData = self.data.groupby(self.clustColumn,sort=True)
		for n, (name,groupData) in enumerate(groupedData):
			if specificAxis is not None:
				if n != subplotId:
					continue
					
			ax = self.axisDict[n] if specificAxis is None else specificAxis
			lineColl = ax.collections
			lineColl[0].set_color(groupData['color'])
			if n == 0 or specificAxis is not None:
				self.plotter.nonCategoricalPlotter.add_legend_for_caetgories_in_scatter(ax,
																self.colorMapDict,categoricalColumns)
		if specificAxis is None:														
			self.add_color_and_size_changes_to_dict('change_color_by_categorical_columns',
																		categoricalColumns)
		
	def change_color_by_numerical_column(self, numericColumn, specificAxis = None, subplotId = None):
		'''
		'''
		cmap = get_max_colors_from_pallete(self.colorMap)
		

		self.data = self.dfClass.join_missing_columns_to_other_df(self.data,id=self.dataID,
																  definedColumnsList=numericColumn)											
		if specificAxis is None:
		
			self.clean_up_saved_size_and_color_changes('color')
				## group data 
		
			if len(numericColumn) > 1:
				if self.plotter.aggMethod == 'mean':
					
					colorData = self.data[numericColumn].mean(axis=1)
				else:
					colorData = self.data[numericColumn].sum(axis=1)
			else:
				colorData = self.data[numericColumn[0]]
			
			scaledData = scale_data_between_0_and_1(colorData) 
			self.data['color'] = [col_c(col.tolist()) for col in cmap(scaledData)]
				
		groupedData = self.data.groupby(self.clustColumn,sort=True)
		for n, (name,groupData) in enumerate(groupedData):
			if specificAxis is not None:
				if n != subplotId:
					continue
			ax = self.axisDict[n] if specificAxis is None else specificAxis
			lineColl = ax.collections
			lineColl[0].set_color(groupData['color'])
		if specificAxis is None:
			self.add_color_and_size_changes_to_dict('change_color_by_numerical_column',numericColumn)		
		
		
	def style_axis(self, ax, groupedData, name, specificAxis = None):
		'''
		'''
		minGroups = groupedData[self.numericalColumns].min().min().min()
		maxGroups = groupedData[self.numericalColumns].max().max().max()
		limitDiff = 0.05 * (maxGroups - minGroups)		
		ax.set_ylim((minGroups - limitDiff,
						 maxGroups + limitDiff))
		ax.set_xlim((-0.5,len(self.numericalColumns)))
		ax.set_xticks(list(range(0,len(self.numericalColumns))))
		if specificAxis is not None or ax.is_last_row():
				ax.set_xticklabels(self.numericalColumns, rotation = 90)
		else:
				ax.set_xticklabels([])
		if specificAxis is None and ax.is_first_col() == False:
				ax.set_yticklabels([])
		else:
				ax.set_ylabel('Value')
		self.plotter.add_annotationLabel_to_plot(ax=ax,
													text = 'Cluster - {}'.format(name))
		
	def replot(self):
		'''
		'''

	def export_selection(self,specificAxis,subplotId):
		'''
		'''
		self.fill_axis(specificAxis,subplotId)	
		#for funcName, column in self.sizeStatsAndColorChanges.items(): 
		#	getattr(self,funcName)(column,specificAxis,subplotId)
		
	def get_data(self):
		'''
		'''
		self.clustColumn = self.plotter.clusterLabels.columns.values.tolist()[0]
		self.clustLabels = self.plotter.clusterLabels[self.clustColumn].unique()
		self.data = self.dfClass.join_missing_columns_to_other_df(self.plotter.clusterLabels,
																 self.dataID,
																 self.numericalColumns)
	def clean_up_saved_size_and_color_changes(self,which = 'color'):
		'''
		Clean up saves from hue color levels.
		'''
		toDelete = []
		for functionName,column in self.sizeStatsAndColorChanges.items(): 
			if which in functionName:
				toDelete.append(functionName)
		if 'change_color_by_categorical_columns' in toDelete:
			self.plotter.delete_legend(self.axisDict[0])
			
		for func in toDelete:
			del self.sizeStatsAndColorChanges[func]			
	
	def update_colorMap(self,newCmap = None):
		'''
		allows changes of color map by the user. Simply updates the color code.
		It also changes the object: self.colorMap so that it will also be used when graph is exported.
		Please note that if you just call the function it will cuase an update, this is particullary useful
		when the user used the interactive widgets to customize the color settings
		'''
		
		if newCmap is not None:
			self.colorMap = newCmap 
			
		for functionName,column in self.sizeStatsAndColorChanges.items(): 
			getattr(self,functionName)(column)  		
	
	def get_raw_colorMapDict(self):
		'''
		'''
		return self.rawColorMapDict
				
	def get_size_color_categorical_column(self, which='change_color_by_categorical_columns'):
		'''
		'''
		return self.sizeStatsAndColorChanges[which]		
	
	def get_current_colorMapDict(self):
		'''
		'''
		return self.colorMapDict
		
		
	def set_user_def_colors(self,categoricalColorDefinedByUser):
		'''
		'''
		self.categoricalColorDefinedByUser = categoricalColorDefinedByUser
				
		