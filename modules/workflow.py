
import tkinter as tk
from tkinter import ttk

from modules.utils import *
from modules import images


_imageWidth_ = 28 #px
_imageHeight_ = 29 #px
class workflowCollection(object):
	
	
	def __init__(self):
		'''
		Inititate Class
		'''
		self.open = False
		self.rebuild = False
		self.justUndone = False
		self.inFront = tk.BooleanVar(value = False)
		self.branches =  OrderedDict()
		self.history = OrderedDict()
		self.endPosition = dict()
		self.imagePositions = dict()
		self.savedEndPos = dict()
		self.lineCoords = dict()
		self.showDetails = OrderedDict()
		self.plotHistory = OrderedDict()
		
		self.openFirstTime = True
		self.itemId = 0
		
		self.get_images()
	
	def add_handles(self, sourceData = None, plotter = None, 
						treeView = None, analyzeData = None):
		'''
		'''
		if sourceData is not None: 
			self.sourceData = sourceData
		if plotter is not None: 
			self.plotter = plotter
		if treeView is not None: 
			self.treeView = treeView
		if analyzeData is not None:
			self.analyzeData = analyzeData
		
	
	def close(self, event = None):
		'''
		Close toplevel.
		'''
		
		
		self.toplevel.destroy()	
		self.open = False
	

	def build_toplevel(self):
		'''
		Create toplevel and bindings
		'''
		
		self.toplevel = tk.Toplevel(bg=MAC_GREY) 
		self.toplevel.wm_title('Workflow') 
		self.toplevel.bind('<Escape>', self.close) 
		if platform == 'MAC':
			self.toplevel.bind('<Command-z>', self.undo)
		else:
			self.toplevel.bind('<Control-z>', self.undo)
			
		self.toplevel.protocol("WM_DELETE_WINDOW", self.close)
		self.center_popup((350,500))
		 
	def change_appearance(self):
		'''
		'''
		if self.inFront.get():
			self.toplevel.attributes('-topmost', 1)
		else:
			self.toplevel.attributes('-topmost', 0)
		
	def build_widgets(self):
		'''
		Create and grid widgets.
		'''
		
		cont = tk.Frame(self.toplevel, background =MAC_GREY)
		cont.pack(fill='both', expand=True)
		cont.grid_columnconfigure(0,weight=1)
		cont.grid_rowconfigure(1,weight=1)
		
		lab1 = tk.Label(cont, text = 'Workflow', **titleLabelProperties)
		lab1.grid(padx=5,pady=15 ,sticky=tk.W)
		
		self.canvas = tk.Canvas(cont, bg= MAC_GREY, bd=0, highlightthickness = 0)
		
		cbInFront = tk.Checkbutton(cont,text='Keep in front', 
					variable = self.inFront, command = self.change_appearance, bg=MAC_GREY)
		cbInFront.grid(row=0,column=3,columnspan=3,sticky=tk.E, padx=3)
		
		hbar = tk.Scrollbar(cont,orient=tk.HORIZONTAL)
		vbar = tk.Scrollbar(cont)
		vbar.config(command=self.canvas.yview)
		hbar.config(command=self.canvas.xview)
		self.canvas.config(xscrollcommand=hbar.set, yscrollcommand=vbar.set,
						 scrollregion=(0,0,5500,6000))		
		
		self.canvas.grid(sticky=tk.NSEW, columnspan=6, padx=5,pady=5)
		vbar.grid(row=1,column=7,sticky=tk.NS+tk.W)
		hbar.grid(sticky=tk.EW, columnspan=6, padx=5,pady=5)
		
		
		self.change_appearance()
		
	def show(self):
		'''
		'''
		if self.open:
			self.toplevel.lift()
			return
			
		self.build_toplevel()
		self.build_widgets()
		self.open = True	
		
		self.restore()
		

	def undo(self,event):
	
		'''
		Handles undoing.
		'''
		if len(self.history) == 0:
			return
		itemId = max(list(self.history.keys()))
		
		if self.history[itemId]['activity'] == 'add_branch':
			
			tagId = 'branch_{}'.format(itemId)
			getattr(self.analyzeData,self.history[itemId]['funcAnalyzeR'])(**self.history[itemId]['argsAnalyzeR'])
			self.canvas.delete(tagId)
			self.canvas.delete(tagId+'_filter')
			if itemId in self.plotHistory:
				branchId = self.plotHistory[itemId]
				
				if branchId in self.branches:
					del self.branches[branchId]
					
			if tagId in self.showDetails:
				del self.showDetails[tagId]
			
			self.remove_item(itemId)
		
		elif self.history[itemId]['activity'] == 'add':
			
			tagId = 'item_{}'.format(itemId)
			branchId = self.history[itemId]['branchId']
			self.sourceData.set_current_data_by_id(branchId)
			

			if 'funcDataR' in self.history[itemId]:
				getattr(self.sourceData,self.history[itemId]['funcDataR'])(**self.history[itemId]['argsDataR'])
			if 'funcTreeR' in self.history[itemId]:
				getattr(self.treeView,self.history[itemId]['funcTreeR'])(**self.history[itemId]['argsTreeR'])			
			if 'funcAnalyzeR' in self.history[itemId]:
				getattr(self.analyzeData,self.history[itemId]['funcAnalyzeR'])(**self.history[itemId]['argsAnalyzeR'])
			if 'funcPlotterR' in self.history[itemId]:
				getattr(self.plotter, self.history[itemId]['funcPlotterR'])(**self.history[itemId]['argsPlotterR'])
			

			if self.branches[branchId][itemId][-1]:
				
				pass # indicates an image. The new end position is calculated as soon
					 # as another task was added. 
				
			else:
			
				self.endPosition[branchId] = self.endPosition[branchId]-40
			
			del self.showDetails[tagId]
			self.remove_item(itemId,branchId)
			self.justUndone  = True	
			
			if self.open: 

				self.canvas.delete(tagId)
						
		
		self.update()		
		
	def clear(self):
		'''
		Deltes all items on canvas.
		'''
		if self.open:
		
			self.canvas.delete('all')
			self.update()
			
		self.branches.clear()
		self.history.clear()
		self.showDetails.clear()
		self.lineCoords.clear()
		self.endPosition.clear()

		self.itemId = 0 
		
	def remove_item(self,itemId, branchId = None):
		'''
		'''
		if itemId in self.history:
		
			del self.history[itemId]
			
		if branchId is not None:
			if itemId in self.branches[branchId]:
				del self.branches[branchId][itemId]
		

	
	def get_images(self):
		'''
		Get images from base64 code. (Module: images.py)
		'''
		self.imageTasks = OrderedDict()
		self.imageTasks['df'], self.imageTasks['filter'], self.imageTasks['calcColumn'], \
		self.imageTasks['deleteRows'], self.imageTasks['deleteColumn'],\
		self.imageTasks['renameColumn'], self.imageTasks['replaceColumn'],\
		self.imageTasks['mergeDfs'], self.imageTasks['appendDfs'],\
		self.imageTasks['subsetDf']  = images.get_workflow_images()		
		
		
		self.imageTasks['boxplot'],self.imageTasks['barplot'],\
		self.imageTasks['violinplot'],self.imageTasks['swarm'],\
		self.imageTasks['pointplot'], self.imageTasks['hclust'],\
		self.imageTasks['corrmatrix'],self.imageTasks['scatter_matrix'],\
		self.imageTasks['line_plot'],\
		self.imageTasks['density'], self.imageTasks['scatter'], self.imageTasks['countplot'] = images.get_workflow_chart_images()

	def get_position(self, type, id = None):
		'''
		'''
		if type == 'branch':
			try:
				numBranches = list(self.branches.keys()).index(id)
			except:
				return 
			y0 = 35
			x0 = 35
			nextPosition = (x0 + numBranches * 120, y0)
			
		return nextPosition



	def add_branch(self, id, fileName, branchInfo = None, sourceBranch = None, addInfo = {}):
		'''
		For each data frame, a new branch is opened. 
		'''
		if self.rebuild == False:
		
			self.branches[id] = OrderedDict()
		
		position = self.get_position('branch',id)
		if position is None:
			return
			
		if id not in self.endPosition:
			
			self.endPosition[id] = position[1] + 65
			

		if self.open:
			
			
			tagId = 'branch_{}'.format(self.itemId)
			self.canvas.create_image(position, 
						image = self.imageTasks['df'], 
						anchor = tk.NW, tag = tagId)
						
			textPosition = (position[0]+29.5, position[1] + 55) #55 = size of image + space.
			
			CreateToolTip(self.canvas,tag_id=tagId, title_ = fileName, text =branchInfo)
			
			self.canvas.create_text(textPosition, text = fileName, 
						font = NORM_FONT, tag = tagId
						)
			
			if sourceBranch is not None:

				self.connect_branches(sourceBranch, id, tagId,image = 'filter', imageInfo = addInfo)
				
				
			self.update()
						
		if self.itemId not in self.branches[id] and self.rebuild == False:
			
			self.branches[id][self.itemId] = ('add_branch',
											  [id,fileName,branchInfo,sourceBranch,addInfo], 
											  False)
			
			self.history[self.itemId] = {'activity':'add_branch',
										 'funcAnalyzeR':'delete_data_frame_from_source',
										'argsAnalyzeR':{'fileIid':id}}
			
			self.plotHistory[self.itemId] = id
			self.itemId += 1



	def delete_branch(self,branchId):
		'''
		'''
		if branchId not in self.branches:
			return
			
		itemDict = self.branches[branchId].copy()
		
		for itemId, details in itemDict.items():
			self.remove_item(itemId) 
			
			del self.plotHistory[itemId]
			
			if self.open:
				if details[0] == 'add_branch':
				
					self.canvas.delete('branch_{}'.format(itemId))
					
				else:
			
					self.canvas.delete('item_{}'.format(itemId))
		
		del self.branches[branchId]
		
		self.update() 		 
			
			
	def update_end_position_by_charts(self,numCharts,isChart,branchId):
		'''
		'''
		if numCharts != 0 and isChart == False:
					
				if numCharts % 4 == 0:
						nGroups = numCharts/4
				else:
						nGroups = int(numCharts/4) + 1
				print(nGroups)
				self.endPosition[branchId] =  self.endPosition[branchId] + (nGroups * _imageHeight_) + 5
			
	def calculate_positions(self, taskName, tagId, branchId, isChart):
		'''
		'''
		if tagId in self.showDetails:
			return self.showDetails[tagId]

		numCharts, tagCharts = self.get_num_of_added_images(branchId)

		if self.justUndone == False:
			self.update_end_position_by_charts(numCharts,isChart,branchId)
		
		y0 = self.endPosition[branchId]

		posBranch = list(self.branches.keys()).index(branchId) + 0.5
		middlePosition = (posBranch * 120, y0)

		if isChart:
		
			posCharts = self.get_image_position(numCharts+1, middlePosition, branchId)
			detailsLine = {}
			
			if numCharts == 0:
				
				detailsLine = {'coords':middlePosition + (middlePosition[0], middlePosition[1]+3),
							  'tag':tagId}
			if numCharts > 0:
				
				self.move_images(tagCharts,posCharts)			
			
			detailsImgs = {'position':posCharts[-1],
						   'image': self.imageTasks[taskName],
						   'anchor': tk.NW, 
						   'tag' : tagId}
			
			self.showDetails[tagId] = [detailsLine,detailsImgs]
			
			return [detailsLine,detailsImgs]
		
		else:
						
			
			lineCoords = middlePosition + (middlePosition[0], middlePosition[1]+10)
			detailsImgs = {'position':(middlePosition[0],middlePosition[1]+12), 
									'image' : self.imageTasks[taskName],
									'anchor' : tk.N, 
									'tag' : tagId}
			detailsLine = {'coords':lineCoords,
						  'tag':tagId}
			
			self.showDetails[tagId] = [detailsLine,detailsImgs]
			self.endPosition[branchId] = self.endPosition[branchId] + 40
			
			return [detailsLine,detailsImgs]
		
		
		

	def add(self, taskName, branchId, addInfo = {}, tagId = None, 
								isChart = False):
		'''
		Adds step to workflow.
		branchId == fileId from data module
		isChart - will arrange chart icons in a more dense way.
		'''
		#taskName = 'chart'
		
		if tagId is None:
			tagId = 'item_{}'.format(self.itemId)
		
		if taskName in self.imageTasks:
			
			plotDetails = self.calculate_positions(taskName, tagId, branchId,isChart)
		
		else:
			return
			
		if self.open:
			
			for n,details in enumerate(plotDetails):
				
				if n == 0 and len(details) != 0:
					
					self.canvas.create_line(details['coords'],tag = details['tag'])				
				
				if n == 1:
				
					self.canvas.create_image(details['position'], image = details['image'],
											tag = details['tag'], anchor = details['anchor'])						
					
				if 'description' in addInfo and 'Activity:' in addInfo['description']:
					CreateToolTip(self.canvas,tag_id = tagId, 
							title_ = 'ID: {} - {}'.format(self.itemId,addInfo['description']['Activity:']), 
							text = addInfo['description'])			
			
				self.update()	
		
			
												
		if isChart or (self.itemId not in self.branches[branchId] and self.rebuild == False):
			
			self.branches[branchId][self.itemId] = ('add',[taskName,branchId,addInfo,tagId],isChart)
			
			self.history[self.itemId] = {'activity':'add',
										'branchId':branchId,
										}
										
			self.history[self.itemId] = merge_two_dicts(self.history[self.itemId],addInfo)
			self.plotHistory[self.itemId] = branchId
			self.itemId += 1
			self.justUndone  = False
		
		
	def get_connecting_lines(self,sourceBranchId,branchId):
		if sourceBranchId+branchId in self.lineCoords:
			return self.lineCoords[sourceBranchId+branchId]
		y0 = int(self.endPosition[sourceBranchId])+0
		posBranchSource = list(self.branches.keys()).index(sourceBranchId) + 0.5	
		middlePositionSource = posBranchSource * 120
		#print(middlePositionSource)
		
		posBranchNew = list(self.branches.keys()).index(branchId) + 0.5	
		middlePositionNew = posBranchNew * 120
		
		lineCoords = [[middlePositionSource,y0,middlePositionSource+60,y0],
					  [middlePositionSource+60,y0,middlePositionSource+60,2],
					  [middlePositionSource+60,1,middlePositionNew ,1],
					  [middlePositionNew,1,middlePositionNew ,35]]
		self.lineCoords[sourceBranchId+branchId] = lineCoords
		return lineCoords 
		
		
	def connect_branches(self, sourceBranchId, branchId, tagId, image = None, imageInfo = {}):
		'''
		Connects two branches. 
		Happens when filtering leads to a new data frame.
		'''
		if sourceBranchId not in self.branches:
			return
		if branchId not in self.branches:
			return
		
		lineCoords = self.get_connecting_lines(sourceBranchId,branchId)		
		for line in lineCoords:
			self.canvas.create_line(*line, fill ="darkgrey", dash = (3,3), tag = tagId)
		
		if image is not None and image in self.imageTasks:
			
			position = (lineCoords[-1][0],3)
			addTag = tagId+'_filter'
			
			self.canvas.create_image(position, anchor = tk.N,
									image = self.imageTasks[image], 
									tag = (addTag))
									
			if 'description' in imageInfo:
				CreateToolTip(self.canvas,tag_id = addTag, 
							title_ = 'ID: {} - {}'.format(self.itemId,imageInfo['description']['Activity:']), 
							text = imageInfo['description'])			
			
			
		
		
	
	def move_images(self,tagIds,positions):
		'''
		Moving images for charts. 
		'''
		for n,tagId in enumerate(tagIds):

			posOld = self.showDetails[tagId][1]['position']
			posNew = positions[n]
			
			if self.open:
			
				imageWidget = [widget for widget in self.canvas.find_withtag(tagId)\
				 if tagId is not None and self.canvas.type(widget)=='image']
				
				self.canvas.move(imageWidget[0],
							 posNew[0]-posOld[0],
							 0)
				itemId = int(tagId.split('_')[-1])
			
				self.imagePositions[itemId] = [self.canvas.coords(imageWidget[0])]
			self.showDetails[tagId][1]['position'] = posNew
		
	def get_image_position(self,nChart,middlePosition, branchId):
		'''
		Calculates image position.
		'''		
		if nChart < 5:
			space = nChart * _imageWidth_	
			leftPosition = middlePosition[0] - space/2
			y = middlePosition[1] + 5
			positions = [(leftPosition + n_ * _imageWidth_ , y) for n_ in range(nChart)]
			
		else:
			totalSpace = 4 * _imageWidth_
			leftPosition = middlePosition[0] - totalSpace/2
			
			y = [middlePosition[1] + 5 + int(n_/4) * _imageHeight_ for n_ in range(nChart)]
			
			positions = [(leftPosition + (n_ - int(n_/4)*4) * _imageWidth_, y[n_]) for n_ in range(nChart)]
					
		return positions			
		
		
			
	def get_num_of_added_images(self,branchId):
		
		#get items and turn them 
		items = list(self.branches[branchId].values())[::-1]
		n = 0
		tagIds = []
		for widget in items:
			if widget[-1]:
				n += 1 
				tagIds.append(widget[1][-1])
			else:
				return n, tagIds[::-1]
		return n, tagIds		
							
			
	
	def restore(self):
	
		self.rebuild = True
		
		for itemId, branchId in self.plotHistory.items():
			
			self.itemId = itemId
			if itemId not in self.branches[branchId]:
				continue
				
			funcArgs = self.branches[branchId][itemId]
			if funcArgs[0] == 'add':
					funcs =  funcArgs[1]+[funcArgs[-1]]
			else:
					funcs = funcArgs[1]
			self.update()
			getattr(self,funcArgs[0])(*funcs)

		self.rebuild = False
		self.itemId += 1
			
	
	def update(self):
		'''
		Updates canvas.
		'''
		if self.open:
			self.canvas.update_idletasks()		
	
	
	def __getstate__(self):
		'''
		Promotes sterilizing of this class (pickle)
		'''		
		state = self.__dict__.copy()
		for list in [ 'canvas','imageTasks','toplevel','inFront','treeView',
									'sourceData','treeView','analyzeData']:#, 'nonCategoricalPlotter', 'plotHistory']:
			if list in state:
				del state[list]
				
		return state			
		
			
	
	
	def center_popup(self,size):
         	'''
         	Casts poup and centers in screen mid
         	'''
	
         	w_screen = self.toplevel.winfo_screenwidth()
         	h_screen = self.toplevel.winfo_screenheight()
         	x = w_screen/2 - size[0]/2
         	y = h_screen/2 - size[1]/2
         	self.toplevel.geometry('%dx%d+%d+%d' % (size + (x, y))) 		
	

	
	