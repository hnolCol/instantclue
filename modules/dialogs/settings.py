import tkinter as tk
from tkinter import ttk    





hierarchClustering = OrderedDict([('Row Dendrogram: ',[('Metric:',['None']+pdist_metric),('Linkage:',linkage_methods)),
								  ('Column Dendrogram: ',['None','Euclidean','Manhatten']),
								  ('Row Cluster Color Palette: ','colorSchemes'),
								  ()]



errorBars = OrderedDict([('Error bar:',['Confidence Interval (0.95)','Standard deviation'])])



class settingsDialog(object):
	
	def __init__(self,plotter,colorHelper):
		
		self.colorSchmes = colorHelper.get_all_color_palettes()
	
	
	
	
	def build_toplevel(self):
	
	
	
	
	def build_widgets(self):
	