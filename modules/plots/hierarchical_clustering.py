"""
	""HIERARCHICAL CLUSTERING""
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
import numpy as np


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import seaborn as sns 
  
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

import fastcluster

from modules.utils import * 


class hierarchichalClustermapPlotter(object):

	def __init__(self,progressClass, dfClass, Plotter,figure,numericColumns = [], plotCorrMatrix = True):
		
		self.exportId = 0
		self.exportAxes = {}
		self.exportYLim = {}
		self.savedLabels = {}
		self.fromSavedSession = False
		
		self.progressClass = progressClass
		self.colorData = pd.DataFrame() 
		self.labelColumn = None
		self.numericColumns = numericColumns
		self.dfClass = dfClass
		self.dataID = dfClass.currentDataFile
		
		self.figure = figure
		
		## this class also plot a correlation matrix
		self.plotCorrMatrix = plotCorrMatrix
		
		##retrieve hclust cmap and dendrogram settings from the Plotter helper.
		## updaten
		self.plotter = Plotter
		cmapClusterMap, self.cmapRowDendrogram, \
		self.cmapColorColumn, self.metricRow, self.metricColumn, \
		self.methodRow, self.methodColumn = self.plotter.get_hClust_settings()
		
		self.cmapClusterMap = get_max_colors_from_pallete(cmapClusterMap) ## to avoid same color over and over again
		
		self.Z_row = None
		self.Z_col = None
		
		self.rectanglesForRowDendro = []
		
		self.labelColumnList = []
		self.colorColumnList = []

		self.get_data()
			

			
		
		
		self.lenDf = len(self.df.index)
		self.axClusterMapXLimits = (0,len(numericColumns))
		
		self.create_cluster_map()
		
	def get_data(self):
		'''
		Retrieve data from dfClass. 
		'''
		
		data = self.dfClass.get_current_data_by_column_list(self.numericColumns )
		self.df = data.dropna(subset=self.numericColumns )
		
		if self.df.empty:
			self.progressClass.update_progressbar_and_label(100,
						'Aborting! NaN Filtering resulted\nin an empty data frame ...')	
			return
								
		self.df_copy = self.df.copy()
		
		if self.plotCorrMatrix:
			self.corrMethod = self.plotter.corrMatrixCoeff
			self.df = self.df.corr(method = self.corrMethod).dropna()
				
	def add_axes_to_figure(self,specificAxis,fig = None, returnAxes = False):
		'''
		'''		
		if fig is None:
			fig = self.figure
		
		if specificAxis is not None:
			##
			positionAxis = specificAxis.get_position()
			x0,x1,y0,y1 = positionAxis.x0,positionAxis.x1,positionAxis.y0,positionAxis.y1
			width = x1-x0
			height = y1-y0
			
			
			addFactorMainWidth =  0
			correctHeight = 210 / 297
			if self.plotCorrMatrix:
				multWidth = 0.75
			else:
				multWidth= 0.55
			#fig.delaxes(specificAxis)
		else:
			x0,y0 = 0.15,0.15
			x1,y1 = 0.95,0.95
			width = x1-x0
			height = y1-y0
			multWidth = 0.4
			correctHeight = 1
			# emperically determined to give almost equal width independent of number of columns 
			
			addFactorMainWidth =  -0.15+len(self.numericColumns) * 0.008 
			
			if width*0.4+addFactorMainWidth > 0.75:
				addFactorMainWidth = width*0.35
				
		if self.plotCorrMatrix and specificAxis is None:
			## to produce a corr matrix in the topleft corner of the graph
			heightMain = height * 0.5
			y0 = 0.4
			height = y1-y0
			
		else:
			heightMain = height * 0.8  	
			
		axRowDendro = fig.add_axes([x0,y0,width* 0.13,heightMain])

		axColumnDendro = fig.add_axes([x0+ width*0.13,
         						y0+heightMain,
         						width*multWidth+addFactorMainWidth,
         						(width* 0.13)*correctHeight])
        						
		axClusterMap	= fig.add_axes([x0+width*0.13,
         							y0,
         							width*multWidth+addFactorMainWidth,
         							heightMain])
         							  
		axLabelColor = fig.add_axes([x0+width*0.13+width*multWidth+addFactorMainWidth+width*0.02,
         							y0,
         							width*0.03,
         							heightMain])	
         							
		axColormap = fig.add_axes([x0,
         							y0+height*0.84,
         							width*0.025,
         							height*0.12])
         
		if self.plotCorrMatrix:
			axColormap.set_title('{} coeff.'.format(self.corrMethod))
         
         
		if returnAxes:
			for axis in fig.axes:
				axis.set_navigate(False)
			return axRowDendro,axColumnDendro,axClusterMap,axLabelColor,axColormap 
      
		else:
			for axis in fig.axes:
				if axis != axClusterMap:
					axis.set_navigate(False)
			self.axRowDendro,self.axColumnDendro,self.axClusterMap,\
									self.axLabelColor,self.axColormap = axRowDendro,axColumnDendro,axClusterMap,axLabelColor,axColormap 
			self.plotter.redraw()

		
	
	def replot(self, updateData = False):
		'''
		'''
		if updateData:
			self.get_data()
		self.progressClass = Progressbar('Hierarchical Clustering ..')
		self.create_cluster_map()
		
	def create_cluster_map(self,specificAxis=None,figure=None):
		'''
		'''
		self.progressClass.update_progressbar_and_label(2,'Starting ...')
		
		self.add_axes_to_figure(specificAxis,figure)
		
		if self.plotCorrMatrix == False: ##if correlation Matrix is used. We dont allow movement of maxD Limits becaause clusters cannot be associated with rows
			self.add_some_bindings()
		
		# get data as numpy arrays
		
		dataRowCluster = self.df[self.numericColumns].values
		dataColCluster = np.transpose(dataRowCluster)
		
		self.progressClass.update_progressbar_and_label(5,'Data collected ...')
		
		if self.metricRow != 'None':
		
			self.rowLinkage, self.rowMaxD = self.cluster_data(dataRowCluster, 
														  self.metricRow, self.methodRow)
			self.Z_row = sch.dendrogram(self.rowLinkage, orientation='left', color_threshold= self.rowMaxD, 
								leaf_rotation=90, ax = self.axRowDendro, no_plot=True)
								
			self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
		
			self.add_dendrogram(self.Z_row,True,self.axRowDendro)
		
			self.progressClass.update_progressbar_and_label(10,'Clustering rows done ...')
		
		else:
			self.axRowDendro.axis('off')
			
		if self.metricColumn != 'None':
			
			columnLinkage,self.colMaxD  = self.cluster_data(dataColCluster, 
														self.metricColumn, self.methodColumn)
		
			self.Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold= self.colMaxD, 
								leaf_rotation=90, ax = self.axColumnDendro, no_plot=True)

			self.progressClass.update_progressbar_and_label(14,'Draw dendrogram ...')
		
			self.add_dendrogram(self.Z_col,False,self.axColumnDendro)
		
			self.progressClass.update_progressbar_and_label(43,'Clustering columns done ...')
		else:
			
			self.axColumnDendro.axis('off')
			
		self.resort_data_frame(self.Z_row,self.Z_col)
		
		self.progressClass.update_progressbar_and_label(64,'Plotting color map ...')
	
		self.colorMesh = self.axClusterMap.pcolormesh(self.df[self.numericColumns].values, cmap = self.cmapClusterMap)
		plt.colorbar(self.colorMesh, cax=self.axColormap)
		
		self.add_maxD_lines()
		
		self.progressClass.update_progressbar_and_label(72,'Adjusting xlimits ...')
		
		self.adjust_axis_limits_and_labels(self.axClusterMap,self.axLabelColor,self.axRowDendro,self.axColumnDendro,self.axColormap)

		
		self.progressClass.update_progressbar_and_label(85,'Draw heatmap ...')
		
		self.plotter.redraw()
		
		self.progressClass.update_progressbar_and_label(100,'Done ...')
		self.progressClass.close()
		
	def get_cluster_number(self,linkage,maxD):
		'''
		'''
		return sch.fcluster(linkage,maxD,'distance')		
		
	
		
	def cluster_data(self, dataFrame, metric, method):
		'''
		'''
		if  metric ==  'euclidean':   
			linkage = fastcluster.linkage(dataFrame, method = method, metric = metric)   
		else:
			distanceMatrix = scd.pdist(dataFrame, metric = metric)
			linkage = sch.linkage(distanceMatrix,method = method)
			del distanceMatrix
			
		maxD = 0.7*max(linkage[:,2])
		return linkage, maxD 
		
	
	def add_dendrogram(self,dendrogram,rotate,ax,create_background = True):
		'''
		Idea is from the great seaborn package.
		'''
		dependent_coord = dendrogram['dcoord']
		independent_coord = dendrogram['icoord']
		max_dependent_coord = max(map(max, dependent_coord))
		

		line_kwargs = dict(linewidths=.45, colors='k')
		if rotate:
			lines = LineCollection([list(zip(x, y))
                                    		for x, y in zip(dependent_coord,
                                            independent_coord)],
                                   			**line_kwargs)
			self.yLimitRow = len(dendrogram['leaves']) * 10
			self.xLimitRow = max_dependent_coord * 1.05
			ax.set_ylim(0, self.yLimitRow)
			ax.set_xlim(0, self.xLimitRow)
			ax.invert_xaxis()

		else:
			lines = LineCollection([list(zip(x, y))
                                    		for x, y in zip(independent_coord,
                                            dependent_coord)],
                                   			**line_kwargs)	
			self.xLimitCol =  len(dendrogram['leaves']) * 10
			self.yLimitCol =  max_dependent_coord * 1.05  
			ax.set_xlim(0, self.xLimitCol)
			ax.set_ylim(0, self.yLimitCol)
			


		ax.add_collection(lines)
		if ax == self.axRowDendro and create_background:
			self.plotter.redraw()
			self.backgroundRow = self.figure.canvas.copy_from_bbox(self.axRowDendro.bbox)
			
	
		
	def resort_data_frame(self,Z_row,Z_col):
		'''
		Reorders data to be plotted as a heatmap
		'''
		if Z_col is not None:
		
			dat = self.df.ix[:,Z_col['leaves']]
			self.numericColumns = [self.numericColumns[idx] for idx in Z_col['leaves']]
			
		else:
		
			dat = self.df
			
		if Z_row is not None:
			self.df = dat.iloc[Z_row['leaves']]
		
		if self.labelColumn is None:
		
			self.labelColumn = self.df.index.tolist()
			

	def adjust_colorLabel_axis(self,numberOfColorColumns, ax = None):
		'''
		'''
		if ax is None:
			ax = self.axLabelColor
		else:
			pass
		bbox = ax.get_position() 
		x0 = bbox.x0
		x1 = bbox.x1
		y0 = bbox.y0
		y1 = bbox.y1
		
		width = x1-x0 
		height = y1-y0
		ax.set_position([x0,y0,width*numberOfColorColumns,height])	
	
	
				
	def adjust_axis_limits_and_labels(self,axClusterMap,axLabelColor,axRowDendro,axColumnDendro,axColormap,export=False):
		'''
		'''
		numbNumericColumns = len(self.numericColumns)
		
		# column names
		axClusterMap.set_xticks(np.linspace(0.5,numbNumericColumns-0.5,num=numbNumericColumns))
		axClusterMap.set_xticklabels(self.numericColumns, rotation=90)
		axClusterMap.xaxis.set_label_position('bottom')
		axClusterMap.yaxis.set_label_position('right')
		axClusterMap.yaxis.tick_right()
		axClusterMap.set_yticklabels([], minor=False)   
		

		#axis to display label/color column, hide axis
		axLabelColor.yaxis.set_label_position('right')
		axLabelColor.set_yticklabels([], minor=False)
		axLabelColor.yaxis.tick_right()
		axLabelColor.axis('off')
		
		##remove xticklabels from dendrogram axes
		axRowDendro.set_xticklabels([])
		axColumnDendro.set_yticklabels([])
		axColumnDendro.set_xticklabels([])
		
		## format colormap 
		axColormap.tick_params(axis=u'y', which=u'both',length=2.3,direction='out')
		ticks = axColormap.get_yticklabels()
		nTicks = len(ticks)
		newTicks = ['' for n in range(nTicks)]		
		for n in np.linspace(start=0,stop=nTicks-1,
								num=3,endpoint=True):
			idx = int(n) 
			newTicks[idx] = ticks[idx].get_text() 
					
		axColormap.set_yticklabels(newTicks)
		
		
		
		#axColormap.yaxis.set_major_locator(mtick.MaxNLocator(5)) 
		
		## adds cluster numbers to dendrogram
		if self.plotCorrMatrix == False:
			self.add_cluster_label(axRowDendro, export = export)
		else:
			axRowDendro.set_yticklabels([])

	
			
			
			
		

	def add_some_bindings(self):
		'''
		'''
		## all available bindings
		
		
		self.y_padding_clust = \
		self.axClusterMap.callbacks.connect('ylim_changed', lambda event:self.on_ylim_change(event))
		self.adjustRowMaxD = \
		self.figure.canvas.mpl_connect('button_press_event', lambda event:self.move_rowMaxD_levels_and_relim(event))
		

	def move_rowMaxD_levels_and_relim(self,event):
		'''
		
		'''
		if event.dblclick:
		
			self.reset_ylimits()
		
		elif event.button != 1:
			return
		elif hasattr(self,'rowMaxDLine') == False:
			return	
		elif self.rowMaxDLine.contains(event)[0]:
			
			self.motion_dendrogram = self.figure.canvas.mpl_connect('motion_notify_event' , 
													lambda event: self.moveRowMaxDLine(event))
			self.release_event = self.figure.canvas.mpl_connect('button_release_event', 
													lambda event: self.redraw_row_dendrogram(event))
                 
               
	def redraw_row_dendrogram(self,event):
		'''
		'''
		self.figure.canvas.mpl_disconnect(self.motion_dendrogram)
		self.figure.canvas.mpl_disconnect(self.release_event) 
		
		self.rowMaxD = event.xdata
		self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
		self.add_cluster_label(self.axRowDendro)
		
		self.plotter.redraw()
		

				
	def moveRowMaxDLine(self,event):
		'''
		'''
		if event.inaxes != self.axRowDendro or event.button != 1:
			self.figure.canvas.mpl_disconnect(self.motion_dendrogram)
			return
			
		self.figure.canvas.restore_region(self.backgroundRow)
		x = event.xdata
		self.rowMaxDLine.set_xdata(x)
		self.axRowDendro.draw_artist(self.rowMaxDLine)
		self.figure.canvas.blit(self.axRowDendro.bbox)
		
   
	def reset_ylimits(self,redraw=True):
		'''
		Resets limit to original scale.
		'''
		if hasattr(self,'yLimitRow') == False:
			
			self.yLimitRow = len(self.df.index)*10
		
		self.axClusterMap.set_ylim(0,self.yLimitRow/10)
		
		if redraw:
		
			self.plotter.redraw()
	
	def reset_xlimits_of_dendro(self,event=None):
		'''
		'''
		if hasattr(self, 'xLimitRow'):
			self.axRowDendro.set_xlim((0, self.xLimitRow))
			self.axRowDendro.invert_xaxis()
		if hasattr(self, 'xLimitCol'):
			self.axColumnDendro.set_xlim((0, self.xLimitCol))
			self.axColumnDendro.set_ylim((0, self.yLimitCol))
			
			
	
	def on_ylim_change(self,event = None):
		'''
		'''
		newYLimits = self.axClusterMap.get_ylim()
		if newYLimits[0] == 0 and newYLimits[1] == 1:
			self.reset_xlimits_of_dendro()
			self.reset_ylimits()
			return
			
		
		newYLimits = [round(x,0) for x in newYLimits]
		numberOfRows = newYLimits[1]-newYLimits[0]	
		
		if self.colorData.empty:
		
			ax = self.axClusterMap
			
		else:
			self.axClusterMap.set_yticklabels([])
			ax = self.axLabelColor
			
			
		if numberOfRows < 60:	
				yTicks = np.linspace(newYLimits[0]+0.5,newYLimits[1]-0.5,
										 num=numberOfRows)
				ax.set_yticks(yTicks)
				 
				ax.set_yticklabels(self.labelColumn[int(newYLimits[0]):int(newYLimits[1])],
												minor=False)
		else:
		
				ax.set_yticklabels([],minor=False)
				
		self.axClusterMap.set_xlim(self.axClusterMapXLimits)
		## sadly very ugly but this is to prevent a slight offset of the other axes.
		## due to rescaling by setting yticks. 
		updatedLim = self.axClusterMap.get_ylim()	
		self.axLabelColor.set_ylim(updatedLim)
		self.axRowDendro.set_ylim((updatedLim[0]*10,updatedLim[1]*10))
		self.plotter.redraw()
		
	def add_cluster_label(self, axRowDendro, export = False):
		'''
		Adds cluster labels to left dendrogram. The multiplication by 10 comes from the values
		given by scipy#s dendrogram function.
		'''
		if hasattr(self,'rowClusterLabel') == False:
			return
		if export == False:
			
			self.clean_up_rectangles()
		
		uniqeClustlabel, countsClust = np.unique(self.rowClusterLabel, return_counts=True)
		countsClust = countsClust.tolist()
		ytickPosition = [(sum(countsClust[0:x])-countsClust[x-1]/2)*10 for x in uniqeClustlabel]
		ytickLabels = ['Cluster {}'.format(cluster) for cluster in uniqeClustlabel]

		axRowDendro.set_yticks(ytickPosition)
		axRowDendro.set_yticklabels(ytickLabels)
		
		colors = sns.color_palette(self.cmapRowDendrogram,uniqeClustlabel.size)
		
		for n,yLimit in enumerate(countsClust):
			if n != 0:
				yLow = sum(countsClust[:n])*10
			else:
				yLow = n
			yLimit = yLimit*10
			rectangle = Rectangle((0,yLow),self.rowMaxD,yLimit,
										facecolor=colors[n],
										alpha=0.75)
			axRowDendro.add_patch(rectangle)
			if export == False:
				self.rectanglesForRowDendro.append(rectangle)
				
		if export == False:							
			self.axLabelColor.set_ylim(self.axClusterMap.get_ylim())
	
	def clean_up_rectangles(self):
		'''
		Deletes all rectangles drawn to indicate clusters
		'''
		for rectangle in self.rectanglesForRowDendro:
			try:
				rectangle.remove()
			except:
				pass
		self.rectanglesForRowDendro = []
				
	def add_maxD_lines(self):
		'''
		Adds lines to indicate lines for limits to distinguish clusters
		'''
		if self.Z_col is not None:	
			self.axColumnDendro.axhline(self.colMaxD , linewidth=1.3, color = 'k')#'#1f77b4')
		if self.Z_row is not None:
			self.rowMaxDLine  = self.axRowDendro.axvline(self.rowMaxD , linewidth=1.3, color = 'k')#'#1f77b4')		
			self.axRowDendro.draw_artist(self.rowMaxDLine)
			self.figure.canvas.blit(self.axRowDendro.bbox)
         		
	
	def add_label_column(self,labelColumnList):
		'''
		'''
		self.labelColumnList = labelColumnList
		
		self.df = self.dfClass.join_missing_columns_to_other_df(self.df,id=self.dataID,
																  definedColumnsList=labelColumnList)
		if len(labelColumnList) == 1 and self.colorData.empty:														  													  
			self.labelColumn = self.df[labelColumnList[0]].values.tolist()
		else:
			self.update_tick_labels_of_rows()
		self.on_ylim_change()
		self.plotter.redraw()

		
	def update_tick_labels_of_rows(self):
		'''
		'''
		combinedLabels = self.labelColumnList + self.colorColumnList
		if len(combinedLabels) == 1:
			self.labelColumn = self.df[combinedLabels[0]].values.tolist()
		elif len(combinedLabels) == 0:
			self.labelColumn = []
		else:	
			tickLabelDf = pd.DataFrame()
			tickLabelDf[combinedLabels] = self.df[combinedLabels].astype(str)
			combinedRowEntries = []
			tickLabelDf[combinedLabels] = tickLabelDf[combinedLabels].applymap(lambda x: x[:15])
			for columnName in combinedLabels[1:]:
				combinedRowEntries.append(self.df[columnName].astype(str).values.tolist())
			
			tickLabelDf.loc[:,'merged'] = self.df[combinedLabels[0]].astype(str).str.cat(combinedRowEntries,sep=', ')

			self.labelColumn = tickLabelDf['merged']
		
			del tickLabelDf
			del combinedRowEntries

        		
	def add_color_column(self,colorColumnList):
		'''
		'''
		self.reset_ylimits(redraw=False)
		self.colorData = pd.DataFrame()
		self.colorColumnList = colorColumnList
		numbInput = len(colorColumnList)
		self.df = self.dfClass.join_missing_columns_to_other_df(self.df,id=self.dataID,
																  definedColumnsList=colorColumnList)
		# adjust color column axis
		self.adjust_colorLabel_axis(numbInput)
				
		self.axLabelColor.axis('on')	
															  
		self.axLabelColor.set_xlim([0,numbInput])			
		
		dataTypes = self.dfClass.get_data_types_for_list_of_columns(colorColumnList)
		
		# get all unique values
		uniqueValuesPerColumn = self.dfClass.get_unique_values(colorColumnList, forceListOutput=True)
		uniqueValuesTotal = np.unique(np.concatenate(uniqueValuesPerColumn))
		
		factorDict = dict(zip(uniqueValuesTotal.tolist(),np.arange(0,uniqueValuesTotal.size)))
		factorDict['-'] = factorDict['nan'] = -1
		#self.colorData[colorColumnList] = self.df.replace(factorDict)
		
		for n,column in enumerate(colorColumnList):
			self.colorData[column] = self.df[column].map(factorDict)
			

		self.draw_color_data(self.axLabelColor,numbInput)
		
		self.update_tick_labels_of_rows()
		## hides the labels or shows them if appropiate (e.g no overlap) 
		self.on_ylim_change()
		self.plotter.redraw()

	def find_index_and_zoom(self,idx):
		'''
		'''
		
		yLimPosition = find_nearest_index(self.df.index,idx)
		
		self.axClusterMap.set_ylim(yLimPosition-5,yLimPosition+5)		
		
		
	def draw_color_data(self,axLabelColor,numbInput):
		'''
		'''
		axLabelColor.pcolormesh(self.colorData, cmap=self.cmapColorColumn)
		
		axLabelColor.set_xticks(np.linspace(0.5,numbInput-0.5,num=numbInput))
		axLabelColor.set_xticklabels(self.colorData.columns, rotation=90)
		axLabelColor.xaxis.set_label_position('top')
		axLabelColor.xaxis.tick_top()
		axLabelColor.tick_params(axis='x', which='major',length=2 ,pad=3)
		
				
	def remove_color_column(self,event = '', label = None):
		'''
		'''
		self.axLabelColor.clear()	
		self.axLabelColor.axis('off')
		self.colorData = pd.DataFrame()
		self.colorColumnList  = []
		self.update_tick_labels_of_rows()
		self.plotter.redraw()
		label.destroy()

	def remove_labels(self, event = '', label = None):
		'''
		'''
		
		self.labelColumnList = []
		self.update_tick_labels_of_rows()
		self.on_ylim_change(event)
		self.plotter.redraw()
		label.destroy()
		
	def change_cmap_of_cluster_map(self,newCmap):
		'''
		Changes colormap of the pcolormesh
		'''
		self.plotter.cmapClusterMap = newCmap
		newCmap = get_max_colors_from_pallete(newCmap)  
		self.colorMesh.set_cmap(newCmap)
		self.axColormap.yaxis.set_major_locator(mtick.MaxNLocator(3))	
		self.cmapClusterMap = newCmap	
		
		
	def export_cluster_number_to_source(self):
		'''
		important: the dataset is not checked again, needs to be done before. (set_current_data_by_id)
		'''
		if hasattr(self,'rowClusterLabel') == False:
			return
		clusterAnnotation = ['Cluster {}'.format(clustNumb) for clustNumb in self.rowClusterLabel]
		# we need to get the original Data again to get the correct index self.df_copy

		newName = self.dfClass.evaluate_column_name('hclust_#')
		annotationDf = pd.DataFrame(clusterAnnotation,columns=[newName],index = self.df_copy.index)
		
		self.dfClass.join_df_to_currently_selected_df(annotationDf)
		self.dfClass.df[newName].fillna('-',inplace=True)
		
		return newName
		
	def export_data_of_corrmatrix(self):
		'''
		Export coor matrix result
		'''
		clusterAnnotation = ['Cluster {}'.format(clustNumb) for clustNumb in self.rowClusterLabel]
		newName = self.dfClass.evaluate_column_name('Cluster ID',self.df.columns.values.tolist())
		annotationDf = pd.DataFrame(clusterAnnotation,columns=[newName],index = self.df_copy.index)
		# save column names for Experiment column annotation
		columnNames = self.df.columns
		# add cluster labels
		outputDf = self.df.join(annotationDf)
		# check if that name is unique and okay
		expName = self.dfClass.evaluate_column_name('Experiment',self.df.columns.values.tolist())
		outputDf.loc[:,expName] = columnNames
		# return the data frame
		return outputDf
		
		
	def export_selection(self, specificAxis = None,figure=None):
		'''
		We repeat here a lot of the function (create_cluster_map) but we believe that 
		it is cleaner like this. 
		
		Function to export the created Clustermap onto another axis (in the given range) 
		'''
		axRowDendro,axColumnDendro,axClusterMap,axLabelColor,axColormap  = self.add_axes_to_figure(specificAxis,figure,returnAxes=True)
		if self.Z_col is not None:	
			self.add_dendrogram(self.Z_col,False,axColumnDendro)
			axColumnDendro.axhline(self.colMaxD , linewidth=1.3, color = 'k')#'#1f77b4')
		else:
			axColumnDendro.axis('off')
		if self.Z_row is not None:
			self.add_dendrogram(self.Z_row,True,axRowDendro,create_background=False)
			axRowDendro.axvline(self.rowMaxD , linewidth=1.3, color = 'k')#'#1f77b4')
		else:	
			axRowDendro.axis('off')
		
		im = axClusterMap.pcolormesh(self.df[self.numericColumns].values, cmap = self.cmapClusterMap)
		plt.colorbar(im, cax=axColormap)
				
		self.adjust_axis_limits_and_labels(axClusterMap,axLabelColor,axRowDendro,axColumnDendro,axColormap,export=True)
		
		if self.fromSavedSession:
			currentYLim = self.exportYLim[self.exportId]
			yticks, ylabels = self.savedLabels[self.exportId]
		else:
			currentYLim = self.axClusterMap.get_ylim()
			if self.colorData.empty == False:
				yticks = self.axLabelColor.get_yticks()
				ylabels = self.axLabelColor.get_yticklabels()
			else:
				yticks = self.axClusterMap.get_yticks()
				ylabels = self.axClusterMap.get_yticklabels()			
		
		if self.colorData.empty == False:
			axLabelColor.axis('on')
			# +0.5 to get enough space for labels! 
			self.adjust_colorLabel_axis(len(self.colorData.columns)+0.5, ax = axLabelColor) 
			self.draw_color_data(axLabelColor,len(self.colorData.columns))
			axLabelColor.set_yticks(yticks)
			axLabelColor.set_yticklabels(ylabels)
		else:
			axClusterMap.set_yticks(yticks)
			axClusterMap.set_yticklabels(ylabels)
			
		## Needed to not get a white box on top of the hclust
			
		for ax in [axClusterMap,axLabelColor]:
			ax.set_ylim(currentYLim)				
		axRowDendro.set_ylim([currentYLim[0]*10,currentYLim[1]*10])
		axesColl = [axRowDendro,axColumnDendro,axClusterMap,axLabelColor,axColormap]
		if self.fromSavedSession == False:
			self.save_export_details(axesColl,
				currentYLim,yticks,ylabels)
		else:
			self.exportAxes[self.exportId] = axesColl
					
	def save_export_details(self,axes,currentYLim,yticks,ylabels):	
		'''
		'''
		self.exportId += 1
		self.exportAxes[self.exportId] = axes
		self.exportYLim[self.exportId] = currentYLim
		if yticks[-1]-yticks[0] < 60:
			self.savedLabels[self.exportId] = [yticks,ylabels]
		else:
			self.savedLabels[self.exportId] = [[],'']
		self.fromSavedSession = False
		
		
		
		
		
	def disconnect_event_bindings(self):
		'''
		disconnects events from canvas
		'''
		if self.plotCorrMatrix == False:
			bindingEvents = [self.adjustRowMaxD]
		
			for event in bindingEvents:
		
				self.figure.canvas.mpl_disconnect(event)
		
	
	def __getstate__(self):
	
		state = self.__dict__.copy()
		if 'progressClass' in state:
			del state['progressClass'] # cannot pickle tkinter app
		if 'backgroundRow' in state:
			del state['backgroundRow'] # cannot pickle canvas.background copy ..
		return state		