import seaborn as sns
import numpy as np

from scipy import interp
from sklearn.metrics import roc_curve, auc

from modules.utils import *

class gridSearchVisualization(object):
	'''
	'''
	def __init__(self,plotter,data):
	
		self.plotter = plotter
		self.figure = plotter.figure
		self.colorMap = self.plotter.get_active_helper().colorMap
		
		self.tooltips = dict()
		self.originalColors = dict()
		self.motionEvent = None
		self.nCVResults = data['nestedCVResults']
		self.rocCurves = data['rocCurveParams']
		self.define_variables()	
		self.prepare_data()
		
		self.create_axes()
		self.fill_axes()
		self.bind_events()
		
	def define_variables(self):
		'''
		Define variables used.
		'''
		self.axisDict = dict() 	
		
		
	def create_axes(self):
		'''
		Adds axes to figure.
		'''
		self.figure.subplots_adjust(right=.95,left=0.15,hspace=0.2,wspace=.2)
		self.axisDict[0] = self.figure.add_subplot(231)
		self.axisDict[1] = self.figure.add_subplot(232)
		self.axisDict[2] = self.figure.add_subplot(233)
		self.axisDict[3] = self.figure.add_subplot(234)
		self.axisDict[4] = self.figure.add_subplot(235)
		self.figure.subplots_adjust(wspace=0.35,hspace=0.28)
	
	def fill_axes(self):
		'''
		'''
		collectMacroRoc = []
		for nSplit, predData in self.rocCurves.items():	
			rocData = predData['roc_curve']
			param = predData['best_params']
			for class_ in self.rocCurves[1]['classes']:
				class_ = str(class_)
				if 'tpr_'+class_ not in rocData:
					continue
				tpr = rocData['tpr_'+class_]
				fpr = rocData['fpr_'+class_]
				aucVal = rocData['AUC_'+class_]
				
			if len(self.rocCurves[1]['classes']) > 2:
				all_fpr = np.unique(np.concatenate([rocData['fpr_'+str(class_)] for class_ in self.rocCurves[1]['classes']]))
				#print(all_fpr)
				mean_tpr = np.zeros_like(all_fpr)
				for inClass in self.rocCurves[1]['classes']:
					inClass = str(inClass)
					#print(interp(all_fpr, rocData['fpr_'+class_], rocData['tpr_'+class_]))
					mean_tpr += interp(all_fpr, rocData['fpr_'+inClass], rocData['tpr_'+inClass])
			
				mean_tpr /= len(self.rocCurves[1]['classes'])
				AUC = round(auc(all_fpr, mean_tpr),2)
				###### PLOT AVERAGED RANK!!!
			else:
				all_fpr = fpr
				mean_tpr = tpr
				AUC = aucVal
				
			paramDetails = self.shorten_params(param)
			   			 
			self.axisDict[0].plot(all_fpr,mean_tpr,
							lw = 0.75,
							label='{}\n (AUC: {},n: {})'.\
							format(paramDetails.replace("'",'').replace('{','').replace('}','')
							,AUC,nSplit))
		self.axisDict[0].set_xlabel('False Positive Rate (1-Specificity)')
		self.axisDict[0].set_ylabel('True Positive Rate (Sensitivity)')
				
		# First aggregate all false positive rates
#all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points



# Finally average it and compute AUC



		leg = self.axisDict[0].legend()
		leg.draggable(state=True, use_blit=True)
		leg._legend_box.align = 'left'
		order = self.nCVResults['params'].unique().tolist()
		
		self.axisDict[1].bar(range(self.testScoreMeans.size),self.testScoreMeans.values)
		self.axisDict[1].set_ylabel('Mean Test Score {}'.format(self.scorePrefix))
		
		
		
		fill_axes_with_plot(plot_type='pointplot',y='mean_test_{}'.format(self.scorePrefix),cmap=self.colorMap,
					hue='params',x='#CV',ax=self.axisDict[2], data = self.nCVResults)
		#sns.pointplot(y='mean_fit_time',hue='params',x='#CV',ax=self.axisDict[3], data = self.nCVResults)
		
		fill_axes_with_plot(plot_type='pointplot',y='mean_fit_time',cmap=self.colorMap,
					hue='params',x='#CV',ax=self.axisDict[3], data = self.nCVResults,
					hue_order = order)
		
		fill_axes_with_plot(plot_type='barplot',y='rank_test_{}'.format(self.scorePrefix),cmap=self.colorMap,
					hue = None, x='params',ax=self.axisDict[4], data = self.nCVResults,
					order = order)
		
		self.remove_legend(self.axisDict[2])
		self.remove_legend(self.axisDict[3])
		self.axisDict[4].set_xticks([])
	
	def remove_legend(self,ax):
		'''
		'''
		leg = ax.get_legend()
		if leg is not None:
			leg.remove()
		
	def prepare_data(self):
  		'''
  		'''
  		## get parameter columns
  		paramColumns = [column for column in self.nCVResults.columns if 'param_' in column]
  		#print(paramColumns) 
  		## group data 
  		groupedResults = self.nCVResults.groupby(paramColumns) 
  		# get combinations in nested cv
  		groupNames = groupedResults.groups.keys() 
  		#print(groupNames)
  		self.scorePrefix = self.rocCurves[1]['scorerPrefix']
  		self.testScoreMeans = groupedResults['mean_test_{}'.format(self.scorePrefix)].mean()
  		#print(self.testScoreMeans)
  	
	def shorten_params(self,param):
		'''
		'''
		param = str(param).replace('FeatureSelection__','').replace('Classifier__','').replace('Pre-Processing__','')
		param = param.replace(',','\n')
		return param
		
	def identify_axis(self,axis):
		'''
		'''
		for id, ax in self.axisDict.items():
			if ax == axis:
				return id
			
		
	def bind_events(self):
		'''
		'''
		self.enterAxis = self.plotter.figure.canvas.mpl_connect('axes_leave_event',self.handle_axis_leave)
		self.exitAxis =  self.plotter.figure.canvas.mpl_connect('axes_enter_event',self.handle_inaxes_events)		
	
	
	
	def handle_axis_leave(self,event):
		'''
		'''
		ax = event.inaxes
		id = self.identify_axis(ax)
		if id is None:
			return
		if id not in self.tooltips:
			return
		self.tooltips[id].reset_original_colors()
		self.tooltips[id].set_invisible()
		
		self.disconnect_motion()
		self.plotter.redraw()
			
			
	def handle_inaxes_motion(self,event):
		'''
		'''
		if event is None:
			return
		ax = event.inaxes
		id = self.identify_axis(ax)
		if id is None:
			return
		if id in self.tooltips:  		
  			self.tooltips[id].evaluate_event(event)
	
	
	def disconnect_motion(self):
		'''
		'''
		if self.motionEvent is not None:
			self.plotter.figure.canvas.mpl_disconnect(self.motionEvent)
			self.motionEvent = None
		
		
			
	
	def handle_inaxes_events(self,event):  	
  		'''
  		'''
  		self.disconnect_motion()
  		ax = event.inaxes
  		id = self.identify_axis(ax)
  		if id is None:
  			return
  		self.currentArtist = None
  		
  		if id not in self.tooltips:
  			
  			
  			artists = dict()
  			colors = dict() 
  			info = dict()  			
  			
  			
  			texts = self.nCVResults['params'].unique().tolist()
  			textsShort = [self.shorten_params(param) for param in texts]
  		
  			if id == 2:
  			
  			
  				axLines = ax.collections
  				for n,line in enumerate(axLines):
  					color = line.get_facecolor().tolist()[0]
  					hex_ = col_c(color)
  					artists[n] = line
  					colors[n] = hex_
  					info[n] = textsShort[n]
  					line.set_facecolor('lightgrey')
  			
  			if id == 4:
  			
  				axPatches = ax.patches
  			
  				axPatches = [patch for patch in axPatches if np.isnan(patch.get_height()) == False]
  				x_pos = [patch.get_x() for patch in axPatches]
  				axPatches = [x for (y,x) in sorted(zip(x_pos,axPatches))]

  			# get names
  			
  			
  				for n,patch in enumerate(axPatches):
  					color = patch.get_facecolor()
  					hex_ = col_c(color)
  					artists[n] = patch
  					colors[n] = hex_
  					info[n] = textsShort[n]
  					patch.set_facecolor('lightgrey')
  			
  			self.originalColors[id] = {'artists':artists,
  									'colors':colors,
  									'texts':info}
  			
  			self.tooltips[id] = chartToolTip(self.plotter,ax,self.originalColors[id])
  			self.tooltips[id].update_background()  	
  			
  		self.plotter.redraw()
  			
  		
  		self.motionEvent =  self.plotter.figure.canvas.mpl_connect('motion_notify_event', 
  																self.handle_inaxes_motion)
  		
  		
	def disconnect_events(self):
  	 
  		'''
  		'''
  		self.plotter.figure.canvas.mpl_diconnect(self.enterAxis)
  		self.plotter.figure.canvas.mpl_diconnect(self.exitAxis)
  		
		


class chartToolTip(object):


	def __init__(self,plotter,ax,artistProps):
		'''
		artistProp - dict. Must have keys : 'artists','colors','texts'. Keys must be all
					the same.
		'''
		
		self.plotter = plotter
		self.r = self.plotter.figure.canvas.get_renderer()
		self.ax = ax
		self.width = self.height = 0
		self.update = True
		self.inactiveColor = 'lightgrey'
		self.currentArtist = None
		self.artistProps = artistProps
		
		self.define_bbox()
		self.define_text()
		self.build_tooltip()
		self.extract_ax_props()
		
				
	
	def update_position(self,event,text):
		'''
		'''
		# get mouse data
		x,y = event.xdata, event.ydata
		## check if new text 
		if self.textProps['text'] != text:
			self.update = True
			
		self.textProps['text'] = text	
		self.textProps['visible'] = True
		self.determine_position(x,y)
		self.tooltip.update(self.textProps)
		self.update_axis()
		

	def evaluate_event(self,event):
		'''
		'''
		n = 0
		for id,artist in self.artistProps['artists'].items():
			if artist.contains(event)[0]:
			
				if hasattr(artist,'set_facecolor'):
					artist.set_facecolor(self.artistProps['colors'][id])
				elif hasattr(artist,'set_color'):
					artist.set_color(self.artistProps['colors'][id])
					
				if self.currentArtist is None or self.currentArtist != artist:
					self.update_background()
				
				self.update_position(event,self.artistProps['texts'][id])
				self.currentArtist = artist
				
				break
			
			elif n == len(self.artistProps['artists'])-1:
				## no match, make tooltip invisible
				artist.set_facecolor(self.inactiveColor)	
				self.currentArtist = None
				self.update_background()
			
			else:
				artist.set_facecolor(self.inactiveColor)	
			n += 1	
	
	def set_all_artists_inactive(self):
		'''
		'''
		for id,artist in self.artistProps['artists'].items():
		
			self.change_color_of_artist(artist,self.inactiveColor)
					
		self.plotter.redraw()		

	def reset_original_colors(self):
		'''
		'''
		for id,artist in self.artistProps['artists'].items():
			self.change_color_of_artist(artist,self.artistProps['colors'][id])
			
					
		self.plotter.redraw()		
	
	
	def change_color_of_artist(self,artist,color):
	
		if hasattr(artist,'set_facecolor'):
				artist.set_facecolor(color)
		elif hasattr(artist,'set_color'):
					artist.set_color(color)
					
	def set_invisible(self,visble = False, update = True):
		'''
		'''
		new = {'visible':visble}
		self.tooltip.update(new)
		if update:
			self.update_axis()
	
	def update_background(self, redraw=True):
		'''
		'''
		if redraw:
			if hasattr(self,'background'):
				self.set_invisible()
			self.plotter.redraw()
		
		self.background =  self.plotter.figure.canvas.copy_from_bbox(self.ax.bbox)
		#self.set_invisible(visble = True)
		
	def update_axis(self):
		'''
		'''
		if hasattr(self,'background'):
			self.plotter.figure.canvas.restore_region(self.background)
			self.tooltip.draw(self.r)
			self.plotter.figure.canvas.blit(self.ax.bbox)
		
	
	def extract_ax_props(self):
		'''
		'''
	
		self.axProps = dict()
		self.axProps['xlim'] = self.ax.get_xlim()
		self.axProps['ylim'] = self.ax.get_ylim()
		self.axProps['xDiff'] = self.axProps['xlim'][1] - self.axProps['xlim'][0]
		self.axProps['yDiff'] = self.axProps['ylim'][1] - self.axProps['ylim'][0]
		
			
	def build_tooltip(self):
		'''
		'''
		self.tooltip = self.ax.text(s ='', bbox=self.bboxProps,**self.textProps)
		self.textProps['text'] = ''
	
	def determine_position(self,x,y):
		'''
		Check how to align the tooltip.
		'''
		if self.update:
			self.extract_text_dim()
			
		xMin,xMax = self.axProps['xlim']
		yMin,yMax = self.axProps['ylim']
		
		width  = self.width
		height = self.height
		
		diff = (xMin-xMax)*0.05	
		
		if x + width > xMax - 0.1*xMax and x > sum(self.axProps['xlim'])/2:
			 self.textProps['ha'] = 'right'
		else:
			self.textProps['ha'] = 'left'
			diff *= -1
		
		if y - height - yMin*0.1 < yMin :
			self.textProps['va'] = 'bottom'
		else:
			self.textProps['va'] = 'top'
		
		self.textProps['x'] = x + diff	
		self.textProps['y'] = y 
		
	def extract_text_dim(self):
		'''
		Extract width and height of a fake text element.
		'''
		fakeText = self.ax.text(0,0,s=self.textProps['text'],bbox=self.bboxProps)
		patch = fakeText.get_window_extent(self.r)
		inv = self.ax.transData.inverted()
		xy0 = list(inv.transform((patch.x0,patch.y0)))
		xy1 = list(inv.transform((patch.x1,patch.y1)))
		self.width = xy1[0]-xy0[0]
		self.height = xy1[1]-xy0[1]	
		fakeText.remove()
		self.update = False
	
	def define_bbox(self):
		'''
		Define bbox
		'''
		self.bboxProps = {'facecolor':'white', 'alpha':0.85,
						 'edgecolor':'darkgrey','fill':True,
						 }
	
	def define_text(self):
		'''
		Define text properties
		'''
		self.textProps = {'x':0,'y':0,'fontname':defaultFont,
						 'linespacing': 1.5,
						 'visible':False,
						 'zorder':20}
		

		
	
	def evaluate_event_for_collection(self,event):
		'''
		'''
		xValue = np.asarray(event.xdata)
		yValue = np.asarray(event.ydata)	
		
		boolXData = np.isclose(self.xData,xValue,atol = 0.01*self.axProps['xDiff'])
		boolYData = np.isclose(self.yData,yValue,atol = 0.01*self.axProps['yDiff'])
		add = np.sum([boolXData,boolYData], axis=0)
		argMax = np.argmax(add == 2)
		if argMax == 0 and add[0] != 2:	
			self.set_invisible()
			return
		textData = self.annotationData[argMax]
		text = get_elements_from_list_as_string(textData).replace(', ','\n')
		self.update_position(event,text)
		
		
	
	def annotate_data_in_collections(self,dfClass,annotationColumnList,
										numericColumns = None, axisId = None,
										scatterCombination = None):
		'''
		'''
		
		helper = self.plotter.get_active_helper()
		dataID = self.plotter.get_dataID_used_for_last_chart()
		data = helper.data 
		if numericColumns  is None:
			numericColumns = helper.numericColumns
		
		#scatter with categories is special since data are separated in
		#different subplots (Axes) 
		if hasattr(helper,'scatterWithCategories'):
			n = 0
			for comb in helper.scatterWithCategories.all_combinations:
				if comb in helper.scatterWithCategories.grouped_keys:
					if axisId == n:
						data =  helper.scatterWithCategories.grouped_data.get_group(comb)
						data = dfClass.join_missing_columns_to_other_df(data,id=dataID,
												 definedColumnsList = annotationColumnList)
						break		
					n+=1
						
		elif hasattr(helper,'linePlotHelper'):
			data = helper.linePlotHelper.data
			data = dfClass.join_missing_columns_to_other_df(data,id=dataID,
												 definedColumnsList = annotationColumnList)
			
			self.yData = data[numericColumns].values
			self.xData = range(len(helper.linePlotHelper.numericColumns))
			self.annotationData = data[annotationColumnList].values
			return			
		
		elif helper._scatterMatrix is not None:
			
			scatMatrix = helper._scatterMatrix
			combination,ax = scatMatrix.axisWithScatter[str(scatterCombination)]
			numericColumns = numericColumns[::-1]
			scatData = scatMatrix.data.dropna(subset=numericColumns)
			data = dfClass.join_missing_columns_to_other_df(scatData,id=dataID,
												 definedColumnsList = annotationColumnList)
	
		
		else: 
		
			data = dfClass.join_missing_columns_to_other_df(data,id=dataID,
												 definedColumnsList = annotationColumnList)	
			
		
		self.xData = data[numericColumns[0]].values
		
		if len(numericColumns) > 1:
			self.yData = data[numericColumns[1]].values 
		else:
			self.yData = np.arange(0,self.xData.size)
			
		self.annotationData = data[annotationColumnList].values
	
	
	def evaluate_event_in_lineCollection(self,event):
		'''
		Check event for Line Collections
		'''		
		xValue = int(event.xdata)
		if abs(xValue - event.xdata) > 0.2:
			self.set_invisible()
			return
		if xValue not in self.xData:
			self.set_invisible()
		else:
			# get column data
			yData = self.yData[:,xValue]
		yValue = np.asarray(event.ydata)
		boolYData = np.isclose(yData,yValue,atol = 0.01*self.axProps['yDiff'])
		arg = np.flatnonzero(boolYData)
		if arg.size < 1:
			self.set_invisible()
		else:
			textData = self.annotationData[arg[0]]
			text = get_elements_from_list_as_string(textData).replace(', ','\n')
			self.update_position(event,text)
			self.plotter.get_active_helper().linePlotHelper.indicate_hover(arg[0])
			

	def annotate_cluster_map(self,dfClass,annotationColumnList):
		'''
		Initiate Tooltip in hierarchical clustering.
		'''
		helper = self.plotter.get_active_helper()
		dataID = self.plotter.get_dataID_used_for_last_chart()
		self.df = helper._hclustPlotter.df
		data = dfClass.join_missing_columns_to_other_df(self.df,id=dataID,
												 definedColumnsList = annotationColumnList)	
		self.annotationData = data[annotationColumnList].values
	
					
	def evaluate_event_in_cluster(self,event):
		'''
		Evaluate event over a hierarchical clustering. 
		'''
		idx = int(event.ydata)
		textData = self.annotationData[idx]
		text = get_elements_from_list_as_string(textData).replace(', ','\n')
		self.update_position(event,text)	
		
		
				
		
				
		
		
		
		
		
		
		
		  				
  		
 