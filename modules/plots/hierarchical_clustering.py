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
import xlsxwriter

from tkinter import filedialog as tf
import tkinter as tk

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick

from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

import math

import seaborn as sns 
  
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as scd

import fastcluster
from modules.dialogs.simple_dialog import simpleListboxSelection

from modules.utils import * 

line_kwargs = dict(linewidths=.45, colors='k')

class hierarchichalClustermapPlotter(object):

	def __init__(self,progressClass, dfClass, Plotter,figure,
				numericColumns = [], plotCorrMatrix = True):
		
		self.exportId = 0
		self.exportAxes = {}
		self.exportYLim = {}
		self.savedLabels = {}
		
		self.polarXTicks = []
		self.clusterLabelsCirc = []
		
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
		self.methodRow, self.methodColumn, self.circulizeDendrogram ,\
		self.showCluster = self.plotter.get_hClust_settings()
		
		self.cmapClusterMap = get_max_colors_from_pallete(cmapClusterMap) ## to avoid same color over and over again
		self.cmapClusterMap.set_bad(color='lightgrey')
		self.Z_row = None
		self.Z_col = None
		
		self.rectanglesForRowDendro = []
		
		self.labelColumnList = []
		self.colorColumnList = []
		self.scatterIds = []

		self.get_data()		
		
		self.lenDf = len(self.df.index)
		self.axClusterMapXLimits = (0,len(numericColumns))
		
		
		if self.circulizeDendrogram :
			self.metricColumn = 'None'
			
		
		self.create_cluster_map()
		
	def get_data(self):
		'''
		Retrieve data from dfClass. 
		'''
		
		data = self.dfClass.get_current_data_by_column_list(self.numericColumns )
		if self.plotCorrMatrix:
			self.corrMethod = self.plotter.corrMatrixCoeff
			self.df = data.corr(method = self.corrMethod).dropna()
		elif self.metricRow == 'None' and self.metricColumn == 'None':
			self.df = data 
		else:
			self.df = data.dropna(subset=self.numericColumns)
			
			
		if self.df.empty:
			self.progressClass.update_progressbar_and_label(100,
						'Aborting! NaN Filtering resulted\nin an empty data frame ...')	
			return
						
		self.df_copy = self.df.copy()
				
	def add_axes_to_figure(self,specificAxis,fig = None, returnAxes = False):
		'''
		'''		
		if fig is None:
			fig = self.figure
		
		if self.circulizeDendrogram :
			if specificAxis is None:
				self.circDAxis = fig.add_subplot(111,polar=True)
			
			return 
		
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

	def x_to_theta(self,x, minX, maxX):
		'''
		'''
		theta = (x-minX)/(maxX-minX) * 1.90 * np.pi
		return theta 
	
	def scale_r_to_1(self,y,maxY):
		'''
		'''
		return abs(y-maxY)/maxY
		
		
	def get_straight_lines(self,points,maxY, minX, maxX):
		'''
		'''
		y1, y2 = self.scale_r_to_1(points[0][1],maxY), self.scale_r_to_1(points[1][1],maxY)
		y3, y4 = self.scale_r_to_1(points[2][1],maxY), self.scale_r_to_1(points[3][1],maxY)
		theta1, theta2 = self.x_to_theta(points[0][0],minX,maxX), self.x_to_theta(points[-1][0],minX,maxX)
		
		return [theta1,theta1],[y1,y2],[theta2,theta2],[y3,y4]

	def find_max_min_in_dn(self,dn):
		'''
		'''
		flattened_listX = [y for x in  dn['icoord'] for y in x]
		flattened_listY = [y for x in  dn['dcoord'] for y in x]
		
		maxX, minX = max(flattened_listX),  min(flattened_listX)
		maxY, minY = max(flattened_listY),  min(flattened_listY)
		
		return maxX,minX,maxY,minY

	def get_connecting_lines(self,points,theta1,theta2,maxY):
		'''
		'''
		y = [self.scale_r_to_1(points[1][1],maxY)] * 50
		x = np.linspace(theta1,theta2,num=50)
		
		return x,y 
   		 
   		
	def circulate_dendrogram(self, ax, dn, create_background = True):
		'''
		'''
		lines = []
		self.endPoints = np.linspace(0,1.90 * np.pi,num = len(dn['leaves']))
		data = [list(zip(x,y)) for x,y in zip(dn['icoord'],dn['dcoord'])]
		maxX,minX,self.maxY,minY = self.find_max_min_in_dn(dn)
		
		for points in data:
		
			x1,y1,x2,y2 = self.get_straight_lines(points,self.maxY,minX,maxX)
			
			xx,yy = self.get_connecting_lines(points,x1[0],x2[0],self.maxY)
			
			lines.append(list(zip(x1,y1)))
			lines.append(list(zip(x2,y2)))
			lines.append(list(zip(xx,yy)))
		
		lines = LineCollection(lines,**line_kwargs)
		ax.add_collection(lines )
		self.progressClass.update_progressbar_and_label(85,'Circulized..')

		ax.grid('off')
		
		ax.set_yticks([])
		if self.plotCorrMatrix:
			ax.set_xticks(self.endPoints)
			ax.set_xticklabels(self.df.columns.values.tolist())
			self.rescale_labels_polar_axis(ax, create = True, 
										   yLimit=1.01)
			self.circDAxis.set_xticks([])
		else:
			ax.set_xticks([])
		ax.set_ylim(0,1)
		
		if self.showCluster:
		
			if create_background:
				self.plotter.redraw()
				self.backgroundRow = self.figure.canvas.copy_from_bbox(self.circDAxis.bbox)		
				
				
		if self.showCluster:
		
			self.progressClass.update_progressbar_and_label(92,'Getting Clusters and plotting..')
			xLimit, maxD = self.draw_cluster_in_circ(self.maxY)
			self.rowMaxDLine = ax.plot(np.linspace(0,xLimit,num=70),[maxD]*70, c = "k", linewidth = 1)[0]


	def draw_cluster_in_circ(self, maxY):
	
			self.remove_clust_labels()
			annotationDf = pd.DataFrame(self.rowClusterLabel,columns=['labels'],index = self.df_copy.index)
			self.uniqueCluster = annotationDf.loc[self.df.index]['labels'].unique()
			valueCounts = annotationDf.loc[self.df.index]['labels'].value_counts()
			self.countsClust = [valueCounts.loc[clustLabel] for clustLabel in self.uniqueCluster]
			uniqeClustlabel, countsClust = np.unique(self.rowClusterLabel, return_counts=True)
			colors = sns.color_palette(self.cmapRowDendrogram,self.uniqueCluster.size)
			
			for n,xLimit in enumerate(self.countsClust):
			
				if n != 0:
					xLow = sum(self.countsClust[:n])*10
				else:
					xLow = n - 0.1
				xLimit = self.x_to_theta(xLow+xLimit*10,0,sum(self.countsClust)*10)
				xLow =  self.x_to_theta(xLow,0,sum(self.countsClust)*10)
				maxD = self.scale_r_to_1(self.rowMaxD,maxY)

				xx = np.linspace(xLow,xLimit,num=70)
					
				polyClust = self.circDAxis.fill_between(xx, 1, maxD,facecolor=colors[n],alpha=0.75)
				self.clusterLabelsCirc.append(polyClust)
			return xLimit, maxD

	def remove_clust_labels(self):
		'''
		'''
		for polyClust in self.clusterLabelsCirc:
			try:
				polyClust.remove()
			except:
				pass
		self.clusterLabelsCirc = []

	def rescale_labels_polar_axis(self, ax, create = False, yLimit = None, 
								  updateText = False):
		'''
		'''		
		if yLimit is None:
			yLimit = ax.get_ylim()[1] + 0.08
		if create:
			if len(self.polarXTicks) != 0:
				for txt in self.polarXTicks:
						try:
							txt.remove()  
						except: 
							pass
						
			self.polarXTicks = []
			#labels = ax.get_xticklabels()
			for n,theta in enumerate(self.endPoints):
				
				rotation = math.degrees(theta)
				if rotation > 90 and rotation < 270:
					ha_ = 'right'
				else:
					ha_ = 'left'
					
				if rotation > 90 and rotation < 270:
					rotation += 180			
					
				txt = ax.text(theta,yLimit, s= self.labelColumn[n],
					ha=ha_,va="center",
					rotation = rotation,
					rotation_mode = 'anchor')
									
				self.polarXTicks.append(txt)
				
			ax.set_xticklabels([])
		else:
			
			for n,txt in enumerate(self.polarXTicks):
				
				theta = self.endPoints[n]
				txt.set_position((theta,yLimit))			
				
					
	def add_outer_grid(self,n,endPoints,ax):
		'''
		Add an outer grid.
		'''
		xx = np.linspace(0,2*np.pi,num=100)
		
		for level in range(n):
			
			ax.plot(xx,[1+0.05*level]*100, color = 'lightgrey', linewidth = 0.2, zorder = 1)
			
			#randInt = np.random.randint(low=0,high = self.endPoints.size, size=70)
			if level == 0:
			
				lines = []
				for p in self.endPoints:
				
					lines.append([(p,1+0.05*level),(p,1+0.05*n)])
			
				lines = LineCollection(lines, colors = 'lightgrey', linewidth = 0.3, zorder = 1)
				ax.add_collection(lines)
			
			#ax.scatter(self.endPoints[randInt],[1+ 0.05*level + 0.025]* 70, sizes = [10])
		
		ax.set_ylim(0,1+0.05*n)
			
	
			
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
		if len(self.df.index) < 2:
			self.progressClass.close()
			tk.messagebox.showinfo('Erro ..','Filtered data frame has less than 2 rows.')
			return
		
		self.progressClass.update_progressbar_and_label(2,'Starting ...')
		
		self.add_axes_to_figure(specificAxis,figure)
		
		if self.plotCorrMatrix == False: 
		##if correlation Matrix is used. We dont allow movement of maxD Limits 
		# becaause clusters cannot be associated with rows
			if self.circulizeDendrogram:
				self.add_some_bindings(self.circDAxis)
			
			else:
				self.add_some_bindings(self.axRowDendro)
		
		# get data as numpy arrays
		
		dataRowCluster = self.df[self.numericColumns].values
		dataColCluster = np.transpose(dataRowCluster)
		
		self.progressClass.update_progressbar_and_label(5,'Data collected ...')
		
		if self.metricRow != 'None' and self.circulizeDendrogram  == False:
		
			self.rowLinkage, self.rowMaxD = self.cluster_data(dataRowCluster, 
														  self.metricRow, self.methodRow)
			if self.rowLinkage is None:
				self.progressClass.close()
				return
				
			self.Z_row = sch.dendrogram(self.rowLinkage, orientation='left', color_threshold= self.rowMaxD, 
								leaf_rotation=90, ax = self.axRowDendro, no_plot=True)
								
			self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
		
			self.add_dendrogram(self.Z_row,True,self.axRowDendro)
		
			self.progressClass.update_progressbar_and_label(10,'Clustering rows done ...')
		
		elif self.circulizeDendrogram  == False:
				
				self.axRowDendro.axis('off')
			
		if self.metricColumn != 'None':
			
			columnLinkage,self.colMaxD  = self.cluster_data(dataColCluster, 
														self.metricColumn, self.methodColumn)
			if columnLinkage is None:
				self.progressClass.close()
				return	
					
			self.Z_col = sch.dendrogram(columnLinkage, orientation='top', color_threshold= self.colMaxD, 
								leaf_rotation=90, ax = self.axColumnDendro, no_plot=True)

			self.progressClass.update_progressbar_and_label(14,'Draw dendrogram ...')
		
			self.add_dendrogram(self.Z_col,False,self.axColumnDendro)
		
			self.progressClass.update_progressbar_and_label(43,'Clustering columns done ...')
			
		elif self.circulizeDendrogram  == False:
			
			self.axColumnDendro.axis('off')
		
		
		
		if self.circulizeDendrogram :	
			
			self.rowLinkage, self.rowMaxD = self.cluster_data(dataRowCluster, 
														  self.metricRow, self.methodRow)
			self.Z_row = sch.dendrogram(self.rowLinkage, orientation='left', 
								color_threshold= self.rowMaxD, 
								no_plot=True)
						
			self.progressClass.update_progressbar_and_label(34,'Dendrogram calculated ..')
			
			self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
			
			self.resort_data_frame(self.Z_row,self.Z_col)
			
			self.progressClass.update_progressbar_and_label(44,'Resorted data frame..')
			self.progressClass.update_progressbar_and_label(65,'Circulize dendrogram..')
			self.circulate_dendrogram(self.circDAxis,self.Z_row)
		
		
		else:
		
			self.resort_data_frame(self.Z_row,self.Z_col)
					
			self.progressClass.update_progressbar_and_label(64,'Plotting color map ...')
			
			if len(self.df.index) < 60 and len(self.df.columns) < 30:
				self.meshKwargs = dict(linewidth = 0.01, linestyle = '-',
					edgecolor = 'k')
			else:
				self.meshKwargs = dict() 
			self.colorMesh = self.axClusterMap.pcolormesh(self.df[self.numericColumns].values, 
													  cmap = self.cmapClusterMap,
													  **self.meshKwargs)
			
			
			self.colorBar = plt.colorbar(self.colorMesh, cax=self.axColormap)
			if self.plotCorrMatrix == False:
				self.axColormap.set_title('n={}'.format(dataRowCluster.shape[0]))
		
			self.add_maxD_lines()
		
			self.progressClass.update_progressbar_and_label(72,'Adjusting xlimits ...')
		
			self.adjust_axis_limits_and_labels(self.axClusterMap,self.axLabelColor,self.axRowDendro,self.axColumnDendro,self.axColormap)
			
			
		
			self.progressClass.update_progressbar_and_label(85,'Draw heatmap ...')
		
		
		self.plotter.redraw()
		
		self.progressClass.update_progressbar_and_label(100,'Done ...')
		self.progressClass.close()
		
	def get_cluster_number(self,linkage,maxD):
		'''
		Returns cluster numbers
		'''
		return sch.fcluster(linkage,maxD,'distance')		
	
		
	def cluster_data(self, dataFrame, metric, method):
		'''
		Clusters the data
		'''
		try:
			if  metric ==  'euclidean':   
				linkage = fastcluster.linkage(dataFrame, method = method, metric = metric)   
			
			else:
			
				distanceMatrix = scd.pdist(dataFrame, metric = metric)
				linkage = sch.linkage(distanceMatrix,method = method)
				del distanceMatrix
		
		except:
				tk.messagebox.showinfo('Error ..','Data could not be clustered. This might be due to rows that contain exactly the same values.')
				return None, None
		
		maxD = 0.7*max(linkage[:,2])
		return linkage, maxD 
		
	
	def add_dendrogram(self,dendrogram,rotate,ax,create_background = True):
		'''
		Idea is from the great seaborn package.
		'''
		dependent_coord = dendrogram['dcoord']
		independent_coord = dendrogram['icoord']
		max_dependent_coord = max(map(max, dependent_coord))
		
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
	
	
				
	def adjust_axis_limits_and_labels(self,axClusterMap,axLabelColor,axRowDendro,
										axColumnDendro,axColormap,export=False):
		'''
		'''
		numbNumericColumns = len(self.numericColumns)
		
		# column names
		axClusterMap.set_xticks(np.linspace(0.5,numbNumericColumns-0.5,num=numbNumericColumns))
		axClusterMap.set_xticklabels(self.numericColumns, rotation=90)

		axClusterMap.xaxis.set_label_position('bottom')
		axClusterMap.yaxis.set_label_position('right')
		axClusterMap.yaxis.tick_right()
		if self.plotCorrMatrix:
			axClusterMap.set_yticks([n+0.5 for n in range(len(self.numericColumns))])
			axClusterMap.set_yticklabels(self.numericColumns)
		else:
			axClusterMap.set_yticklabels([], minor=False)   
		

		#axis to display label/color column, hide axis
		axLabelColor.yaxis.set_label_position('right')
		axLabelColor.set_yticklabels([], minor=False)
		axLabelColor.yaxis.tick_right()
		try: #matplotlib 3.0
			axLabelColor.axis(False)
		except:
			axLabelColor.axis('off')
		
		##remove xticklabels from dendrogram axes
		axRowDendro.set_xticklabels([])
		axColumnDendro.set_yticklabels([])
		axColumnDendro.set_xticklabels([])
		
		## format colormap 
		self.format_colorMap_ticks(axColormap)
				
		## adds cluster numbers to dendrogram
		if self.showCluster and self.plotCorrMatrix == False:
			self.add_cluster_label(axRowDendro, export = export)
		else:
			axRowDendro.set_yticklabels([])
	
	def format_colorMap_ticks(self,axColormap):
		'''
		Tick of color map formatting.
		'''
		axColormap.tick_params(axis=u'y', which=u'both',length=2.3,direction='out')
		yMin, yMax = axColormap.get_ylim() 
		yTickTlabels = [return_readable_numbers(x) for x in np.linspace(yMin,yMax,num=3,endpoint=True)]
		
		self.colorBar.set_ticks(np.linspace(yMin,yMax,num=3,endpoint=True))
		self.colorBar.set_ticklabels(yTickTlabels)
		
		
	def add_some_bindings(self,ax):
		'''
		'''
		## all available bindings
		if self.circulizeDendrogram == False:
			self.y_padding_clust = \
			self.axClusterMap.callbacks.connect('ylim_changed', 
							lambda event:self.on_ylim_change(event))
			
			if hasattr(self,'axRowDendro'):
				self.axRowDendro.callbacks.connect('ylim_changed',self.avoid_row_miss_scale)
			
			if hasattr(self,'axColumnDendro'):
				self.axColumnDendro.callbacks.connect('ylim_changed',self.avoid_column_miss_scale)
		
		self.adjustRowMaxD = \
		self.figure.canvas.mpl_connect('button_press_event', 
							lambda event:self.move_rowMaxD_levels_and_relim(event,ax))
		
	def avoid_row_miss_scale(self,event):
		'''
		'''
		ymin, ymax = self.axRowDendro.get_ylim()
		if ymax - ymin == 1:
			self.reset_xlimits_of_row_dendro()
	
	def avoid_column_miss_scale(self,event):
		'''
		'''
		if hasattr(self,'yLimitCol') == False:
			return
		if self.axColumnDendro.get_ylim()[1] != self.yLimitCol:
			self.reset_xlimits_of_col_dendro()		
			
	def move_rowMaxD_levels_and_relim(self,event,ax):
		'''
		
		'''
		if event.dblclick:
		
			self.reset_ylimits()
		
		elif event.button != 1:
			return
		elif hasattr(self,'rowMaxDLine') == False:
			return	
			
		elif event.xdata is None:
			return 
		elif event.ydata is None and self.circulizeDendrogram:
			return
		elif self.rowMaxDLine.contains(event)[0]:
			
			self.motion_dendrogram = self.figure.canvas.mpl_connect('motion_notify_event' , 
													lambda event: self.moveRowMaxDLine(event,ax))
			self.release_event = self.figure.canvas.mpl_connect('button_release_event', 
													lambda event: self.redraw_row_dendrogram(event,ax))
                 
               
	def redraw_row_dendrogram(self,event,ax):
		'''
		'''
		self.figure.canvas.mpl_disconnect(self.motion_dendrogram)
		self.figure.canvas.mpl_disconnect(self.release_event) 
		if self.circulizeDendrogram:
			if event.ydata is not None:
				self.rowMaxD = (1 - event.ydata) * self.maxY
				self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
			else:
				maxD = self.scale_r_to_1(self.rowMaxD,self.maxY)
				self.rowMaxDLine.set_ydata([maxD]*70)
			self.draw_cluster_in_circ(self.maxY)
		else:
			
			if event.xdata is not None:
				self.rowMaxD = event.xdata
				self.rowClusterLabel = self.get_cluster_number(self.rowLinkage,self.rowMaxD)
			else:
				self.rowMaxDLine.set_xdata(self.rowMaxD)
			
			self.add_cluster_label(ax)
		
		self.plotter.redraw()
		if self.circulizeDendrogram:
			self.backgroundRow = self.figure.canvas.copy_from_bbox(self.circDAxis.bbox)	 
		

				
	def moveRowMaxDLine(self,event = None,ax = None):
		'''
		'''
		if event.inaxes != ax or event.button != 1:
			self.figure.canvas.mpl_disconnect(self.motion_dendrogram)
			return
			
		self.figure.canvas.restore_region(self.backgroundRow)
		if self.circulizeDendrogram:
		
			self.rowMaxDLine.set_ydata([event.ydata]*70)
			
		else:
			x = event.xdata
			self.rowMaxDLine.set_xdata(x)
		ax.draw_artist(self.rowMaxDLine)
		self.figure.canvas.blit(ax.bbox)
		
   
	def reset_ylimits(self,redraw=True):
		'''
		Resets limit to original scale.
		'''
		if hasattr(self,'yLimitRow') == False:
			
			self.yLimitRow = len(self.df.index)*10
		
		self.axClusterMap.set_ylim(0,self.yLimitRow/10)
		
		if redraw:
		
			self.plotter.redraw()
	
	def reset_xlimits_of_row_dendro(self,event=None):
		'''
		'''
		if hasattr(self, 'xLimitRow'):
			self.axRowDendro.set_xlim((0, self.xLimitRow))
			self.axRowDendro.set_ylim((0,self.yLimitRow))
			self.axRowDendro.invert_xaxis()
		
	def reset_xlimits_of_col_dendro(self,event=None):
		'''
		'''
		if hasattr(self, 'xLimitCol'):
			self.axColumnDendro.set_xlim((0, self.xLimitCol))
			self.axColumnDendro.set_ylim((0, self.yLimitCol))
			
			
	
	def on_ylim_change(self,event = None):
		'''
		Function handling actions on ylim change.
		'''
		if self.circulizeDendrogram:
			return
		if hasattr(self,'axClusterMap') == False:
			return
			
		newYLimits = self.axClusterMap.get_ylim()
		if newYLimits[0] == 0 and newYLimits[1] == 1:
			self.reset_xlimits_of_row_dendro()
			self.reset_ylimits()
			return
			
		newYLimits = [round(x,0) for x in newYLimits]
		numberOfRows = newYLimits[1]-newYLimits[0]	
		
		if self.colorData.empty:
			ax = self.axClusterMap
		else:
			self.axClusterMap.set_yticklabels([])
			ax = self.axLabelColor
			
		self.update_linestyle(numberOfRows)
		if numberOfRows < 55:	
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
	
	
	def update_linestyle(self, numberOfRows, mesh = None):
		'''
		Update linestyle of Quadmesh to show nice borders if appropiate.
		'''
		if hasattr(self,'colorMesh') == False and mesh is None:
			return
		
		elif mesh is None:
			
			mesh = self.colorMesh
			
		if numberOfRows < 60 and len(self.numericColumns) < 30:
			props = dict(linewidth = 0.01, linestyle = '-',
					edgecolor = 'k')
			mesh.update(props)
		else:
			props = dict(linewidth = 0, linestyle = '-',
					edgecolor = 'k')
			mesh.update(props)
			
	def add_cluster_label(self, axRowDendro, export = False):
		'''
		Adds cluster labels to left dendrogram. The multiplication by 10 comes from the values
		given by scipy#s dendrogram function.
		'''
		if hasattr(self,'rowClusterLabel') == False:
			return
		if export == False:
		
			self.clean_up_rectangles()
		# get index of original data and sort the df according to the resorting done.
		
		annotationDf = pd.DataFrame(self.rowClusterLabel,columns=['labels'],index = self.df_copy.index)
		self.uniqueCluster = annotationDf.loc[self.df.index]['labels'].unique()
		valueCounts = annotationDf.loc[self.df.index]['labels'].value_counts()
		self.countsClust = [valueCounts.loc[clustLabel] for clustLabel in self.uniqueCluster]

		ytickPosition = [(sum(self.countsClust[0:n+1])-valueCounts.loc[x]/2)*10 for n,x in enumerate(self.uniqueCluster)]
		ytickLabels = ['Cluster {}'.format(cluster) for cluster in self.uniqueCluster]

		axRowDendro.set_yticks(ytickPosition)
		axRowDendro.set_yticklabels(ytickLabels)
		
		colors = sns.color_palette(self.cmapRowDendrogram,self.uniqueCluster.size)
		
		for n,yLimit in enumerate(self.countsClust):
			if n != 0:
				yLow = sum(self.countsClust[:n])*10
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
		if self.circulizeDendrogram or (len(labelColumnList) == 1 and self.colorData.empty):														  													  
			self.labelColumn = self.df[labelColumnList[0]].values.tolist()
		else:
			self.update_tick_labels_of_rows()
		if self.circulizeDendrogram:
			self.circDAxis.set_xticks(self.endPoints)
			self.circDAxis.set_xticklabels(self.labelColumn)
			
			self.rescale_labels_polar_axis(self.circDAxis, create = True, 
										   )
			self.circDAxis.set_xticks([])
		else:
			self.on_ylim_change()
			
		self.plotter.redraw()
		
				
	def update_tick_labels_of_rows(self):
		'''
		'''
		combinedLabels = list(set(self.labelColumnList + self.colorColumnList))
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
		if self.circulizeDendrogram  == False:
		
			self.reset_ylimits(redraw=False)
		
		self.colorData = pd.DataFrame()
		self.colorColumnList = colorColumnList
		numbInput = len(colorColumnList)
		self.df = self.dfClass.join_missing_columns_to_other_df(self.df,id=self.dataID,
																  definedColumnsList=colorColumnList)		
		
		if self.circulizeDendrogram :
			self.legendParams = OrderedDict()
			
			dataTypes = self.dfClass.get_data_types_for_list_of_columns(colorColumnList)
			# get numericColumns 
			#uniqueValuesPerColumn = self.dfClass.get_unique_values(colorColumnList, forceListOutput=True)
			#nTotal = sum([x.size for x in uniqueValuesPerColumn])
			
			self.df['row_idx_InstantClue'] = range(len(self.df.index))
			
			for id in self.scatterIds:
				m = id
				m.remove()
			scatterIds = []
			if len(self.figure.legends) != 0:
				self.figure.legends[0].remove()
					
			n = 0
			
			for n,column in enumerate(self.colorColumnList):
				
				if dataTypes[n] == 'object':
					

					uniqueValues = self.dfClass.get_unique_values(column)
					for uniqueVal in uniqueValues:
			
						subset = self.df[self.df[column] == uniqueVal]
						rowIdx = subset['row_idx_InstantClue']
						yValue = 1.025 + 0.05 * n 				
						scat = self.circDAxis.scatter(self.endPoints[rowIdx],
								[yValue] * len(rowIdx.index), 
								sizes = [10], zorder=4, label = uniqueVal)
						scatterIds.append(scat)
						self.legendParams[uniqueVal] = n
						n+=1
					self.figure.legend(scatterIds,uniqueValues,'upper right')
						
				else:
					
					cmap = get_max_colors_from_pallete(self.cmapColorColumn)
					scaledData = scale_data_between_0_and_1(self.df[column]) 
					colors = cmap(scaledData)
					yValue = 1.025 + 0.05 * n 	
					
					self.circDAxis.scatter(self.endPoints,[yValue] * self.endPoints.size, 
						facecolors = colors, sizes = (scaledData) * 10 + 2, zorder=4)
					self.legendParams[column] = n
					n += 1
				
			#else:
			
			self.add_outer_grid(n,self.endPoints,self.circDAxis)
			self.circDAxis.set_ylim(0,yValue+0.025)
			self.rescale_labels_polar_axis(self.circDAxis,False,yValue+0.08)
			
			
			
						
		
		else:
		
			# get all unique values
			uniqueValuesPerColumn = self.dfClass.get_unique_values(colorColumnList, forceListOutput=True)
			uniqueValuesTotal = np.unique(np.concatenate(uniqueValuesPerColumn))
			self.factorDict = dict(zip(uniqueValuesTotal.tolist(),np.arange(0,uniqueValuesTotal.size)))
			self.factorDict['-'] = self.factorDict['nan'] = -1
			#self.colorData[colorColumnList] = self.df.replace(factorDict)	
				
			for n,column in enumerate(colorColumnList):
				self.colorData[column] = self.df[column].map(self.factorDict)
			# adjust color column axis		
						
			self.adjust_colorLabel_axis(numbInput)
			self.axLabelColor.axis('on')	
			self.axLabelColor.set_xlim([0,numbInput])			
		
			self.draw_color_data(self.axLabelColor,numbInput)
		
			self.update_tick_labels_of_rows()
			## hides the labels or shows them if appropiate (e.g no overlap) 
			self.on_ylim_change()
			
		self.plotter.redraw()
		if self.circulizeDendrogram:
			self.backgroundRow = self.figure.canvas.copy_from_bbox(self.circDAxis.bbox)			

	def find_index_and_zoom(self,idx):
		'''
		'''
		yLimPosition = find_nearest_index(self.df.index,idx)
		if hasattr(self,'axRowDendro') and hasattr(self,'xLimitRow'):	
			yData = (yLimPosition+0.5)*10
			if hasattr(self,'indicatorLines'):
				self.indicatorLines[0].set_data([0, self.xLimitRow],
												[yData,yData])
												
			else:
			
				self.indicatorLines  = self.axRowDendro.plot([0, self.xLimitRow],
							[yData,yData],color='red',
							linestyle='-',
							linewidth = 1.2)
								
		self.axClusterMap.set_ylim(yLimPosition-15,yLimPosition+15)		
		self.on_ylim_change()
		
	def draw_color_data(self,axLabelColor,numbInput):
		'''
		'''
		self.colorDataMesh = axLabelColor.pcolormesh(self.colorData, cmap=self.cmapColorColumn,**self.meshKwargs)
		
		axLabelColor.set_xticks(np.linspace(0.5,numbInput-0.5,num=numbInput))
		axLabelColor.set_xticklabels(self.colorData.columns, rotation=90)
		axLabelColor.xaxis.set_label_position('top')
		axLabelColor.xaxis.tick_top()
		axLabelColor.tick_params(axis='x', which='major',length=2 ,pad=3)
		
				
	def remove_color_column(self,event = '', label = None):
		'''
		'''
		if self.circulizeDendrogram:
			tk.messagebox.showinfo('Error.','Cannot remove color from circularized dendrogram. Please replot.')
			return
		self.axLabelColor.clear()	
		try:
			self.axLabelColor.axis(False)
		except:
			self.axLabelColor.axis('off')
		self.colorData = pd.DataFrame()
		self.colorColumnList  = []
		self.update_tick_labels_of_rows()
		self.plotter.redraw()
		if label is not None:
			label.destroy()

	def remove_labels(self, event = '', label = None):
		'''
		'''
		
		self.labelColumnList = []
		self.update_tick_labels_of_rows()
		if self.circulizeDendrogram and hasattr(self,'circDAxis'):
			for txt in self.polarXTicks:
				txt.update({'visible':False})
			self.polarXTicks = []
		else:
			self.on_ylim_change(event)
		self.plotter.redraw()
		if label is not None:
			label.destroy()
		
	def change_cmap_of_cluster_map(self,newCmap):
		'''
		Changes colormap of the pcolormesh
		'''
		if hasattr(self,'colorMesh') == False:
			return
		self.plotter.cmapClusterMap = newCmap
		newCmap = get_max_colors_from_pallete(newCmap)  
		self.colorMesh.set_cmap(newCmap)
		self.cmapClusterMap = newCmap	
		self.format_colorMap_ticks(self.axColormap)
		
		
	def export_cluster_number_to_source(self):
		'''
		important: the dataset is not checked again, needs to be done before. (set_current_data_by_id)
		'''
		if hasattr(self,'rowClusterLabel') == False:
			return
	
		# we need to get the original Data again to get the correct index self.df_copy

		newName = self.dfClass.evaluate_column_name('hclust_#')
		annotationDf = pd.DataFrame(['Clust_{}'.format(x) for x in self.rowClusterLabel],columns=[newName],index = self.df_copy.index)
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
			axColumnDendro.set_axis_off()
		if self.Z_row is not None:
			self.add_dendrogram(self.Z_row,True,axRowDendro,create_background=False)
			axRowDendro.axvline(self.rowMaxD , linewidth=1.3, color = 'k')#'#1f77b4')
		else:	
			axRowDendro.set_axis_off()
		
		im = axClusterMap.pcolormesh(self.df[self.numericColumns].values, 
								cmap = self.cmapClusterMap,**self.meshKwargs)

		plt.colorbar(im, cax=axColormap)
		if self.plotCorrMatrix == False:
				axColormap.set_title('n={}'.format(self.df.shape[0]))
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
			axLabelColor.axis(True)
			# +0.5 to get enough space for labels! 
			self.adjust_colorLabel_axis(len(self.colorData.columns)+0.5, ax = axLabelColor) 
			self.draw_color_data(axLabelColor,len(self.colorData.columns))
			axLabelColor.set_yticks(yticks)
			axLabelColor.set_yticklabels(ylabels)
		else:
			axClusterMap.set_yticks(yticks)
			axClusterMap.set_yticklabels(ylabels)
			
		newYLimits = [round(x,0) for x in currentYLim]
		numberOfRows = newYLimits[1]-newYLimits[0]	
		self.update_linestyle(numberOfRows, mesh = im)			
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
		
		

	def save_data_to_excel(self):
		'''
		'''		
		pathSave = tf.asksaveasfilename(initialdir=path_file,
                                        title="Choose File",
                                        filetypes = (("Excel files","*.xlsx"),),
                                        defaultextension = '.xlsx',
                                        initialfile='hClust_export')
		if pathSave == '' or pathSave is None:
			return
       
		selectableColumns = self.dfClass.get_columns_of_df_by_id(self.dataID)
		columnsNotUsed = [col for col in selectableColumns if col not in self.df.columns]
		selection = []
		if len(columnsNotUsed) != 0:
			dialog = simpleListboxSelection('Select column to add from the source file',
         		data = columnsNotUsed)   		
			selection = dialog.selection
		
		workbook = xlsxwriter.Workbook(pathSave)
		worksheet = workbook.add_worksheet()
		nColor = 0
		currColumn = 0
		colorSave = {}
		clustRow = 0
		
		progBar = Progressbar(title='Excel export')
		
		colorsCluster = sns.color_palette(self.cmapRowDendrogram,self.uniqueCluster.size)[::-1]
		countClust_r = self.countsClust[::-1]
		uniqueClust_r = self.uniqueCluster[::-1]
		progBar.update_progressbar_and_label(10,'Writing clusters ..')
		for clustId, clustSize in enumerate(countClust_r):
			for n in range(clustSize):
				cell_format = workbook.add_format() 
				cell_format.set_bg_color(col_c(colorsCluster[clustId]))
				worksheet.write_string(clustRow + 1,
					0,'Cluster_{}'.format(uniqueClust_r[clustId]), 
					cell_format)
				clustRow += 1
		
		progBar.update_progressbar_and_label(20,'Writing column headers ..')
		
		for n ,colHead in enumerate(['Clust_#'] +\
			 self.numericColumns + self.colorData.columns.tolist()  + \
			 ['Cluster Index','Data Index'] +\
			 self.labelColumnList + selection):
			 
			worksheet.write_string(0, n, colHead)		 
		
		colorArray = self.colorMesh.get_facecolors()#np.flip(,axis=0)	
		totalRows = int(colorArray.shape[0]/len(self.numericColumns))
		progBar.update_progressbar_and_label(22,'Writing cluster map data ..')

		for nRow in range(totalRows):
			for nCol in range(len(self.numericColumns)):
				c = colorArray[nColor].tolist()
				if str(c) not in colorSave:
					colorSave[str(c)] = col_c(c)
				cell_format = workbook.add_format({'align': 'center',
                                     			   'valign': 'vcenter',
                                     			   'border':1,
                                     			   'bg_color':colorSave[str(c)]}) 
				worksheet.write_number(totalRows - nRow ,nCol + 1,self.df.iloc[nRow,nCol], cell_format)
				nColor += 1
				
		worksheet.set_column(1,len(self.numericColumns),3)
		worksheet.freeze_panes(1, 0)
		progBar.update_progressbar_and_label(37,'Writing color data ..')

		if len(self.colorData.columns) != 0:
			currColumn = nCol + 1
			colorFac_r = dict((v,k) for k,v in self.factorDict.items())
			colorArray = self.colorDataMesh.get_facecolors()
			nColor = 0		
			totalRows = int(colorArray.shape[0]/len(self.colorData.columns))	
			for nRow in range(totalRows):
				for nCol in range(len(self.colorData.columns)):
					c = colorArray[nColor].tolist()
					if str(c) not in colorSave:
						colorSave[str(c)] = col_c(c)
						
					cellInt = self.colorData.iloc[nRow,nCol]
					cellStr = str(colorFac_r[cellInt])
					
					cell_format = workbook.add_format({
                                     			   'border':1,
                                     			   'bg_color':colorSave[str(c)]}) 
                                     			   
					worksheet.write_string(totalRows - nRow, nCol + 1 + currColumn , cellStr, cell_format)
					nColor += 1
					
		currColumn = nCol + 1 + currColumn	
						
		for n,idx in enumerate(np.flip(self.df.index,axis=0)):
			worksheet.write_number(n+1,currColumn+1,n+1)
			worksheet.write_number(n+1,currColumn+2,idx + 1)	
			
		progBar.update_progressbar_and_label(66,'Writing label data ..')
		if len(self.labelColumnList) != 0:
			for nRow, labelStr in enumerate(self.labelColumn):
				worksheet.write_string(totalRows-nRow,currColumn+3,str(labelStr))
			
		progBar.update_progressbar_and_label(77,'Writing additional data ..')
		
		df = self.dfClass.join_missing_columns_to_other_df(self.df, self.dataID, definedColumnsList = selection) 
		df = df[selection]
		dataTypes = dict([(col,df[col].dtype) for col in selection])
		if len(selection) != 0:
			for nRow in range(totalRows):
				data = df.iloc[nRow,:].values
				for nCol in range(len(selection)):
					cellContent = data[nCol]
					if dataTypes[selection[nCol]] == 'object':
						worksheet.write_string(totalRows-nRow, currColumn+4+nCol,str(cellContent))
					else:
						try:
							worksheet.write_number(totalRows-nRow, currColumn+4+nCol,cellContent)
						except:
							#ignoring nans
							pass

		workbook.close()
		progBar.update_progressbar_and_label(100,'Done..')
		progBar.close()
	
		
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