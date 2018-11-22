"""
	""CLUSTERING""
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
import tkinter as tk
from tkinter import ttk             
import tkinter.simpledialog as ts
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import pandas as pd
import numpy as np
import sklearn.cluster as skClust
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabaz_score

from modules.utils import *
from modules.dialogs.VerticalScrolledFrame import VerticalScrolledFrame

availableMethods = ['k-means','MiniBatch-Kmeans','DBSCAN',
					'Spectral Clustering','Birch','Affinity Propagation',
					'Agglomerative Clustering']

classDict = {'k-means':skClust.KMeans,'DBSCAN':skClust.DBSCAN,'Birch':skClust.Birch,'Spectral Clustering':skClust.SpectralClustering,
			'MiniBatch-Kmeans':skClust.MiniBatchKMeans,'Agglomerative Clustering':skClust.AgglomerativeClustering,
			'Affinity Propagation':skClust.AffinityPropagation}


dbscanWidgets = OrderedDict([('eps',['0.5','The maximum distance between two samples for them to be considered as in the same neighborhood.']),
					 ('min_samples',['5','The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.']),
					 ('metric',['euclidean',pdist_metric,'The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.']),
					 ('algorithm',['auto',['auto', 'ball_tree', 'kd_tree', 'brute'],'The algorithm to be used by the NearestNeighbors module to compute pointwise distances and find nearest neighbors. See NearestNeighbors module documentation for details.']),
					 ('leaf_size',['30',list(range(5,50,5)),'Leaf size passed to BallTree or cKDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem']),
					 ('n_jobs',['-2','The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.']),
					 ])

kmeansWidgets = OrderedDict([('n_clusters',['8','The number of clusters to form as well as the number of centroids to generate.']),
					 ('init',['k-means++',['k-means++','random'],'Method for initialization, defaults to ‘k-means++’:\n‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.\n‘random’: choose k observations (rows) at random from data for the initial centroids.']),
					 ('n_init',['10','Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.']),
					 ('n_jobs',['1','The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.']),
					 ])

AffinityPropagationWidgets = OrderedDict([('damping',['0.5','Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). This in order to avoid numerical oscillations when updating these values.']),
					 ('convergence_iter',['15','Number of iterations with no change in the number of estimated clusters that stops the convergence.']),
					 ])

MiniBatchKMeansWidgets = OrderedDict([('n_clusters',['8','The number of clusters to form as well as the number of centroids to generate.']),
					 ('init',['k-means++',['k-means++','random'],'Method for initialization, defaults to ‘k-means++’:\n‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence. See section Notes in k_init for more details.\n‘random’: choose k observations (rows) at random from data for the initial centroids.']),
					 ('batch_size',['100','Size of the mini batches.']),
					 ('n_init',['10','Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.']),
					 ])

spectralClustWidgets = OrderedDict([('n_clusters',['8','The dimension of the projection subspace.']),
					 ('n_init',['10','Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.']),
					 ('affinity',['rbf',['rbf','nearest_neighbors', 'sigmoid', 'polynomial', 'poly', 'linear', 'cosine'],'If a string, this may be one of ‘nearest_neighbors’, ‘rbf’ or one of the kernels supported by sklearn.metrics.pairwise_kernels. Only kernels that produce similarity scores (non-negative values that increase with similarity) should be used. This property is not checked by the clustering algorithm']),
					 ('n_neighbors',[10,'Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity=rbf']),
					 ('n_jobs',['-2','The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.']),
					 
					 ])
birchWidgets = OrderedDict([('n_clusters',['8','The dimension of the projection subspace.']),
					 ('threshold',['0.5','The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold. Otherwise a new subcluster is started. Setting this value to be very low promotes splitting and vice-versa.']),
					 ('branching_factor',['50','Maximum number of CF subclusters in each node. If a new samples enters such that the number of subclusters exceed the branching_factor then that node is split into two nodes with the subclusters redistributed in each. The parent subcluster of that node is removed and two new subclusters are added as parents of the 2 split nodes.']),
					 ])

agglomerativeWidgets  = OrderedDict([('n_clusters',['3','The number of clusters to find.']),
					 ('affinity',['euclidean',['euclidean','l1','l2','manhatten','cosine'],'Metric used to compute the linkage. Can be “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or ‘precomputed’. If linkage is “ward”, only “euclidean” is accepted..']),
					 ('linkage',['ward',['ward','complete','average'],'Which linkage criterion to use. The linkage criterion determines which distance to use between sets of observation. The algorithm will merge the pairs of cluster that minimize this criterion. Ward minimizes the variance of the clusters being merged. Average uses the average of the distances of each observation of the two sets. complete or maximum linkage uses the maximum distances between all observations of the two sets']),
					 ])

widgetCollection = {'DBSCAN':dbscanWidgets,'k-means':kmeansWidgets,
					'Affinity Propagation':AffinityPropagationWidgets,
					'MiniBatch-Kmeans':MiniBatchKMeansWidgets,
					'Spectral Clustering':spectralClustWidgets,
					'Birch':birchWidgets, 'Agglomerative Clustering':agglomerativeWidgets}


class clusterAnalysisCollection(object):
	'''
	'''
	def __init__(self):
		
		self.clusterClasses = OrderedDict() 
		self.clusterProperties = OrderedDict()
	
	def add_new_cluster_analysis(self, name, clusterClass, clusterLabels, silhouetteScore, calinskiScore, numericColumns):
		'''
		'''
		self.clusterClasses[name] = clusterClass
		self.clusterProperties[name] = {'ClusterLabels':clusterLabels,
										'silhouette-score':silhouetteScore,
										'calinski-score':calinskiScore,
										'numericColumns':numericColumns}
		
	@property
	def get_all_performed_clusterAnalysis(self):
		
		return self.clusterClasses
	
	def get_names_of_analysis(self):
		
		return list(self.clusterClasses.keys())
		
		
class clusteringDialog(object):
	
	
	def __init__(self, dfClass, plotter, dataTreeview, clusterCollection, widgetHandler,
						numericColumns = [], initialMethod = 'k-means', cmap = 'Blues'):
		'''
		'''
		self.connectCenter = tk.BooleanVar(value=True)
		self.addClusterLabels = tk.BooleanVar(value=True)
		self.collectCurrentWidgets = dict()
		self.settingDict = dict()
		
		self.initialMethod = initialMethod
		self.numericColumns  = 	numericColumns	
		self.cmap = cmap
		
		self.dfClass = dfClass
		self.dataID = plotter.get_dataID_used_for_last_chart()
		self.dfClass.set_current_data_by_id(self.dataID)
		self.data = dfClass.get_current_data_by_column_list(numericColumns)
		self.data = self.data.dropna()
		
		self.plotter = plotter
		self.widgetHandler = widgetHandler
		
		self.dataTreeview = dataTreeview
		self.clusterCollection = clusterCollection
		self.build_popup()
		self.add_widgets_to_toplevel()
	
	def close(self):
		'''
		Closes the toplevel.
		'''
		
		self.toplevel.destroy()
               
			
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Clustering') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 550
		h= 380
		self.toplevel = popup
		self.center_popup((w,h))	

	def add_widgets_to_toplevel(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(2,weight=1)
		
		self.contClustMethod = tk.Frame(self.cont,background=MAC_GREY)
		self.contClustMethod.grid(row=5,column=0,columnspan = 4, sticky= tk.NSEW)
		self.contClustMethod.grid_columnconfigure(1,weight=1)
		
		labelTitle = tk.Label(self.cont, text = 'Unsupervised clustering algorithms', 
                                     **titleLabelProperties)
		labelHelp = tk.Label(self.cont, text ='We are using the skilearn.cluster module and therefore names \nof parameters and descriptions '+
												'are similiar. The sklearn webpage has\na brilliant overview of the available cluster methods, advantages and disadvantages.',
												justify=tk.LEFT, bg=MAC_GREY)
									
		clusterMethod = tk.Label(self.cont, text = 'Clustering Algorithm: ', bg=MAC_GREY)
		comboBoxClusters = ttk.Combobox(self.cont, values = availableMethods)
		comboBoxClusters.insert(tk.END, self.initialMethod)
		comboBoxClusters.bind('<<ComboboxSelected>>', self.new_algorithm_selected)
		
		labelSklearnWebsite = tk.Label(self.cont, text = 'sklearn.cluster webpage', **titleLabelProperties)
		labelSklearnWebsite.bind('<Button-1>', self.openWebsite)
		
		labelSettings = tk.Label(self.cont, text = 'Cluster analysis initial settings', bg=MAC_GREY)
				
		buttonPerformCluster = ttk.Button(self.cont, text = 'Done', command = self.perform_analysis)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		
		labelTitle.grid(row=0,padx=5,sticky=tk.W,pady=5, columnspan=3)
		labelSklearnWebsite.grid(row=0, column= 3, sticky=tk.E,pady=5)
		labelHelp.grid(row=1, column= 0, columnspan=4, sticky=tk.W,pady=5,padx=5)
		
		clusterMethod.grid(row=2,column=0, sticky=tk.E,padx=5,pady=3)
		comboBoxClusters.grid(row=2,column=1, columnspan= 4, sticky=tk.EW,padx=(20,40),pady=3)
		labelSettings.grid(row=3,column=0,padx=5,pady=2,columnspan=2, stick=tk.W)
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=4,columnspan=4,sticky=tk.EW,padx=(6,4))
		
		self.create_clusterMethod_specific_widgets(self.initialMethod)
		
		ttk.Separator(self.cont,orient=tk.HORIZONTAL).grid(row=6,columnspan=4,sticky=tk.EW,padx=(4,6))
		
		
		checkbuttonConnect = ttk.Checkbutton(self.cont, text = 'Connect data with cluster center', variable = self.connectCenter)
		CreateToolTip(checkbuttonConnect, text = 'This is only applicable if cluster method is one out of [k-means,Affinity Propagation,MiniBatch-Kmeans]. Cluster centers will be connected with individual points.')
		
		checkbuttonConnect.grid(row = 7, column = 0, columnspan = 3, sticky = tk.E)
		buttonPerformCluster.grid(row=8,column=0,padx=3,pady=5,sticky=tk.W)
		closeButton.grid(row=8,column=3,padx=3,pady=5,sticky=tk.E)
		

	def create_clusterMethod_specific_widgets(self, method = 'DBSCAN'):
		'''
		'''
		widgetInfoDict = widgetCollection[method]
		self.collectCurrentWidgets.clear()
		
		n = 0
		for label, widgetInfo in widgetInfoDict.items():
			labelWidget = tk.Label(self.contClustMethod, text = '{} :'.format(label), bg=MAC_GREY)
			labelWidget.grid(sticky=tk.E, padx=5,column= 0, row = n)
			
			if isinstance(widgetInfo[1],str):
				entry = ttk.Entry(self.contClustMethod) 
				entry.insert(tk.END,widgetInfo[0])
				entry.grid(sticky=tk.EW,column= 1, row = n, padx= (20,40))
				self.collectCurrentWidgets[label] = entry
			else:
				combobox = ttk.Combobox(self.contClustMethod, values = widgetInfo[1], exportselection=0)
				combobox.insert(tk.END,widgetInfo[0])
				combobox.grid(sticky=tk.EW,column= 1, row = n, padx= (20,40))
				self.collectCurrentWidgets[label] = combobox
				
			CreateToolTip(labelWidget,text = widgetInfo[-1])
			n += 1
			
	def new_algorithm_selected(self, event):
		'''
		'''
		combo = event.widget
		newAlgorithm = combo.get()
		if newAlgorithm == self.initialMethod:
			return
			
		if newAlgorithm in availableMethods:
		
			for widget in self.contClustMethod.winfo_children():
				widget.destroy()
			self.create_clusterMethod_specific_widgets(newAlgorithm)
			self.initialMethod = newAlgorithm
			
	def perform_analysis(self):
		'''
		'''
		try:
			self.extract_current_settings()  # defines self.settingDict
		except:
			tk.messagebox.showinfo('Error ..','While extracting the settings, an error occured.')
			return
		if 'n_clusters' in self.settingDict:
			try:
				if float(self.settingDict['n_clusters']) > 1:
					pass
				else:
					tk.messagebox.showinfo('Error..','n_clusters must be > 1.')
					return
			except:
				tk.messagebox.showinfo('Error..','Could not interpret setting n_clusters.')
				return
		try:
			clusterClass = classDict[self.initialMethod](**self.settingDict)
		except:
			tk.messagebox.showinfo('Error..','Initializing cluster estimator failed. Check settings and data.')
		try:
			clusterLabels = clusterClass.fit_predict(self.data[self.numericColumns])
		except ValueError as e:
			tk.messagebox.showinfo('Error..','Initializing the clustering an error occured:\n\n' + str(e))
			return

		if self.initialMethod == 'DBSCAN':
			uniqueLabels = np.unique(clusterClass.labels_).size
			if uniqueLabels == 1:
				tk.messagebox.showinfo('Error ..','DBSCAN - All data were assigned as noise. Reconsider settings.')
				return
		columnName = self.dfClass.evaluate_column_name('Clust-Labels: {}'.format(self.initialMethod))
		classLabels = pd.DataFrame(clusterClass.labels_, 
								   index=self.data.index, columns = [columnName], dtype = 'object')
		classLabels[columnName] = classLabels[columnName].astype(str)
		
		## calculate silhouette and calinksi score	   
		silhouetteScore = silhouette_score(self.data[self.numericColumns], clusterClass.labels_, metric = 'euclidean')
		calinskiScore = calinski_harabaz_score(self.data[self.numericColumns], clusterClass.labels_)
		scoreDict = {'Silhouette':silhouetteScore, 'Calinski': calinskiScore}
		self.plotter.update_cluster_anaylsis_evalScores(scoreDict, classLabels)
		self.dfClass.set_current_data_by_id(self.dataID)
		self.data = self.data.join(classLabels)
		self.data[columnName].fillna(self.dfClass.replaceObjectNan)
		## add data to dfClass
		self.dfClass.join_df_to_currently_selected_df(classLabels)	
		# sort values
		self.dfClass.sort_columns_by_value(columnName)
		# replace nan in data (happens if missing values are present in source data)
		self.dfClass.fill_na_in_columnList(columnName)
		
		## plot the cluster result
		self.plotter.initiate_chart(numericColumns = self.numericColumns,
									categoricalColumns = [], selectedPlotType = 'cluster_analysis',
									colorMap=self.cmap, redraw = False)	
		## give clusters color
		self.plotter.nonCategoricalPlotter.change_color_by_categorical_columns(columnName, adjustLayer = False)

		## connect cluster if desired for some methods
		#if self.initialMethod in ['k-means','Affinity Propagation','MiniBatch-Kmeans'] and self.connectCenter.get():
		#	self.plot_cluster_centers(clusterClass, columnName)
				
		self.plotter.redraw()
		## 
		self.clusterCollection.add_new_cluster_analysis(columnName.split(' :')[-1], # this looks overly complicated but it ensures that a name cannot be present twice.
													    clusterClass,clusterClass.labels_,
													    silhouetteScore,
													    calinskiScore,
													    self.numericColumns)
		
		self.dataTreeview.add_list_of_columns_to_treeview(id = self.dfClass.currentDataFile,
															dataType = 'object',
															columnList = [columnName])
		self.widgetHandler.clean_frame_up()
		self.widgetHandler.create_widgets(plotter = self.plotter)
		del clusterClass
		
	def extract_line_segments(self, clusterCentersDf, numericColumns, clusterColumn):
		'''
		'''
		totalNCluster = len(clusterCentersDf.index)
		lineSegments = []
		clustCenterCoord =  clusterCentersDf[numericColumns].apply(tuple, axis=1)
		for clusterNum in range(totalNCluster):
			## cluster center coordinates
			clustCenterCoords = clustCenterCoord.iloc[clusterNum]
			## get cluster data
			clustSubset = self.data[self.data[clusterColumn].astype(str).str.contains('^{}$'.format(clusterNum))]
			clustSubset.loc[:,'coords'] = clustSubset[numericColumns].apply(tuple, axis=1)
			line  = [[clustCenterCoords,value] for value in clustSubset['coords'].values]
			lineSegments.append(line)
			
		return lineSegments
		
	
	def plot_cluster_centers(self,clusterClass, columnName):
		'''
		Calculate line collection to connect cluster center with data/scatter points
		_____Depracted!_____ 
		'''
		clustCenters = clusterClass.cluster_centers_[:,:len(self.numericColumns)]
		clusterSeq = pd.DataFrame(clustCenters, columns = self.numericColumns)
		lineSegments = self.extract_line_segments(clusterSeq, self.numericColumns[:2], columnName)
		colors = self.plotter.nonCategoricalPlotter.get_current_colorMapDict()
		return
		for n,segment in enumerate(lineSegments):
			color = colors[str(n)]
			self.plotter.nonCategoricalPlotter.add_line_collection(segment, colors=color, zorder = 1)
						
	def extract_current_settings(self):
		'''
		Extract current settings.
		'''
		if len(self.settingDict) > 0:
			self.settingDict.clear() 
		
		for key, widget in self.collectCurrentWidgets.items():
			
			stringEntry = widget.get()
			if key in ['eps','damping','threshold']:
				value = float(stringEntry)
			elif key in ['min_samples','leaf_size','n_init','convergence_iter',
						'batch_size','n_neighbors','branching_factor','n_clusters','n_jobs']:
				value = int(float(stringEntry))
			else:
				value = stringEntry
			self.settingDict[key] = value

	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))             
	
	
	
	def openWebsite(self,event):
		'''
		'''
		webbrowser.open_new(r"http://scikit-learn.org/stable/modules/classes.html#module-sklearn.cluster")	
	




class predictCluster(object):
	
	def __init__(self, clusterCollection, dfClass, dataTreeview, numericColumns = []):
	
		self.clusterCollection = clusterCollection
		self.dfClass = dfClass 
		self.dataTreeview = dataTreeview 
		
		if len(numericColumns) == 0:
			self.numericColumns  = self.dataTreeview.columnsSelected
		else:
			self.numericColumns = numericColumns
			
		self.availableClusterClasses = clusterCollection.get_names_of_analysis()
		
		if len(self.availableClusterClasses) == 0:
			
			tk.messagebox.showinfo('Error ..','Could not find any performed cluster analysis.')
			return
		
		self.build_popup()
		self.add_widgets_to_toplevel() 
		
		
		
		
	def close(self):
		'''
		Closes the toplevel.
		'''
		
		self.toplevel.destroy()
         
         			
	def build_popup(self):
		'''
		Builds the toplevel to put widgets in 
		'''
        
		popup = tk.Toplevel(bg=MAC_GREY) 
		popup.wm_title('Predict clusters') 
		popup.protocol("WM_DELETE_WINDOW", self.close)
		w = 530
		h= 420
		self.toplevel = popup
		self.center_popup((w,h))			

	def add_widgets_to_toplevel(self):
		'''
		'''
		self.cont= tk.Frame(self.toplevel, background = MAC_GREY) 
		self.cont.pack(expand =True, fill = tk.BOTH)
		self.cont.grid_columnconfigure(1,weight=1)
		self.cont.grid_rowconfigure(5,weight=1)
		
		self.contClust = tk.Frame(self.cont,background=MAC_GREY)
		self.contClust.grid(row=5,column=0,columnspan = 4, sticky= tk.NSEW)
		self.contClust.grid_columnconfigure(1,weight=1)	
		
		labelTitle = tk.Label(self.cont, text = 'Predict cluster classes', 
                                     **titleLabelProperties)
                                     
		labelInfo = tk.Label(self.cont, text = 'Select available cluster analysis for prediction.'+
											   '\nThe result will be an additional column in the \ndata treeview per selected predictor indicating class labels.',
											   justify=tk.LEFT, bg=MAC_GREY)
											   
		labelWarning = tk.Label(self.cont, text = 'Warning: If column names do not match exactly only the order of input matters..',**titleLabelProperties)
                                     
                                     
		self.create_widgets_for_clustClasses() 
		
		applyButton = ttk.Button(self.cont, text = 'Predict', command = self.perform_prediction)
		closeButton = ttk.Button(self.cont, text = 'Close', command = self.close)
		labelTitle.grid(row=0,padx=5,sticky=tk.W,pady=5, columnspan=3)
		labelInfo.grid(row=1,padx=5,sticky=tk.W,pady=5, columnspan=4)
		labelWarning.grid(row=2,padx=5,sticky=tk.W,pady=5, columnspan=4)
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(row=3,columnspan=4,sticky=tk.EW, padx=1,pady=3)
		
		ttk.Separator(self.cont, orient = tk.HORIZONTAL).grid(row=6,columnspan=4,sticky=tk.EW, padx=1,pady=3)
		applyButton.grid(row=7,column=0,padx=3,pady=5)
		closeButton.grid(row=7,column=3,padx=3,pady=5, sticky=tk.E)
        

	def create_widgets_for_clustClasses(self):
		'''
		'''
		self.clust_cbs_var = dict() 
		vertFrame = VerticalScrolledFrame(self.contClust)
		vertFrame.pack(expand=True,fill=tk.BOTH)
		for clustClass in self.availableClusterClasses:
		
			varCb = tk.BooleanVar(value = False) 
			textInfo = self.clusterCollection.clusterClasses[clustClass].get_params()
			infoDict = self.clusterCollection.clusterProperties[clustClass]
			columnsInClustClass = infoDict['numericColumns']
			
			cb = ttk.Checkbutton(vertFrame.interior, text = clustClass, variable = varCb) 
			self.clust_cbs_var[clustClass] = varCb			
			
			if 'Spectral Clustering' in clustClass:
				cb.configure(state = tk.DISABLED)
				title_ = '== Spectral Clustering does not support predictions. =='
			elif 'Agglomerative' in clustClass:
				cb.configure(state = tk.DISABLED)
				title_ = '== Agglomerative Clustering does not support predictions. =='
			elif len(columnsInClustClass) != len(self.numericColumns):
				cb.configure(state=tk.DISABLED) 
				title_ = ' == Number of selected columns/features does NOT match the\nnumber of columns/features used in cluster class creation! == '
			else:		
				title_ =  'Cluster settings\nColumns: {}\n\nSilhouette-Score: {}\nCalinski-Score: {}'.format(get_elements_from_list_as_string(columnsInClustClass), 
																		round(infoDict['silhouette-score'],3), round(infoDict['calinski-score'],2))
			CreateToolTip(cb, text = str(textInfo).replace("'",''), title_ = title_)
			cb.grid(sticky=tk.W, column=0,padx=3,pady=3)	

		
		vertFrame.grid_rowconfigure(len(self.availableClusterClasses)+1,weight=1)
    	
	def perform_prediction(self):
		'''
		'''
		dataToPredict = self.dfClass.df[self.numericColumns].dropna()
		predictLabelsColumns = []
		for key, variable in self.clust_cbs_var.items():
			if variable.get():
				infoDict = self.clusterCollection.clusterProperties[key]
				resortedColumns = self.resort_columns_if_same_name(infoDict['numericColumns'])
				
				if resortedColumns is not None:
					dataToPredict = dataToPredict[resortedColumns]	
				
				clustLabels = self.clusterCollection.clusterClasses[key].predict(dataToPredict.as_matrix())
				columnName = 'Predict {}'.format(key)
				columnName = self.dfClass.evaluate_column_name(columnName)
				dataToPredict.loc[:,columnName] = clustLabels
				dataToPredict[columnName] = dataToPredict[columnName].fillna(self.dfClass.replaceObjectNan).astype(str)
				predictLabelsColumns.append(columnName)
		
		dataToPredict[predictLabelsColumns].fillna(self.dfClass.replaceObjectNan)
		self.dfClass.join_df_to_currently_selected_df(dataToPredict[predictLabelsColumns])
		self.dataTreeview.add_list_of_columns_to_treeview(self.dfClass.currentDataFile, 
														 'object', predictLabelsColumns)		
		
		tk.messagebox.showinfo('Done ..','Cluster prediction done. Prediction was added to the data treeview.')
		del dataToPredict
    
    		
	def resort_columns_if_same_name(self,predictorColumns):
		'''
		'''
		if all(column in predictorColumns for column in self.numericColumns):		
			indices = [predictorColumns.index(column) for column in self.numericColumns]
			
			numericColumnsSorted = [y for x,y in zip(indices,self.numericColumns)]
			return numericColumnsSorted
		else:
			return None    		

    	                                 		
	def center_popup(self,size):

         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry("%dx%d+%d+%d" % (size + (x, y)))     		
		
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	



