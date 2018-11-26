
from modules.utils import *
import matplotlib.patches as patches
	
class annotateScatterPoints(object):
	'''
	Adds an annotation triggered by a pick event. Deletes annotations by right-mouse click
	and moves them around.
	Big advantage over standard draggable(True) is that it takes only the closest annotation to 
	the mouse click and not all the ones below the mouse event.
	'''
	def __init__(self,Plotter,ax,data,labelColumns,
					numericColumns,madeAnnotations,selectionLabels):
		
		self.plotter = Plotter
		self.ax = ax
		self.data = data
		self.numericColumns = numericColumns
		self.textAnnotationColumns = labelColumns
		self.eventOverAnnotation = False
		
		self.selectionLabels = selectionLabels
		self.madeAnnotations = madeAnnotations
		
		self.on_press =  self.plotter.figure.canvas.mpl_connect('button_press_event', lambda event:  self.onPressMoveAndRemoveAnnotions(event))
		self.on_pick_point =  self.plotter.figure.canvas.mpl_connect('pick_event', lambda event:  self.onClickLabelSelection(event))
	
	def disconnect_event_bindings(self):
		'''
		'''
		
		self.plotter.figure.canvas.mpl_disconnect(self.on_press)
		self.plotter.figure.canvas.mpl_disconnect(self.on_pick_point)
		
	def onClickLabelSelection(self,event):
		'''
		Drives a matplotlib pick event and labels the scatter points with the column choosen by the user
		using drag & drop onto the color button..
		'''
		if hasattr(event,'ind'):
			pass
		else:
			return
		## check if the axes is used that was initiated
		## allowing several subplots to have an annotation (PCA)
		if event.mouseevent.inaxes != self.ax:
			return
		
		if event.mouseevent.button != 1:
			self.plotter.castMenu = False
			return
					
		ind = event.ind
		## check if we should label
		xyEvent =  (event.mouseevent.xdata,event.mouseevent.ydata)
		self.find_closest_and_check_event(xyEvent,event.mouseevent)
		if self.eventOverAnnotation:
			return
		
		selectedData = self.data.iloc[ind]
		key = selectedData.index.values.item(0)
		clickedData = selectedData.iloc[0]
		
		
		if key in self.selectionLabels: ## easy way to check if that row is already annotated
			return
		
		xyDataLabel = tuple(clickedData[self.numericColumns])
		if len(self.textAnnotationColumns) != 1:
			textLabel = str(clickedData[self.textAnnotationColumns]).split('Name: ')[0] ## dangerous if 'Name: ' is in column name
		else:
			textLabel = str(clickedData[self.textAnnotationColumns].iloc[0])
		ax = self.ax
		xLimDelta,yLimDelta = xLim_and_yLim_delta(ax)
		xyText = (xyDataLabel[0]+ xLimDelta*0.02, xyDataLabel[1]+ yLimDelta*0.02)
		annotObject = ax.annotate(s=textLabel,xy=xyDataLabel,xytext= xyText, ha='left', arrowprops=arrow_args)
		
		self.selectionLabels[key] = dict(xy=xyDataLabel, s=textLabel, xytext = xyText)
		self.madeAnnotations[key] = annotObject
		
		self.plotter.redraw()
	
	def addAnnotationFromDf(self,dataFrame, redraw = True):
		'''
		'''
		ax = self.ax

		for rowIndex in dataFrame.index:
			if rowIndex not in self.data.index:
				continue
			textData = self.data[self.textAnnotationColumns].loc[rowIndex]
			textLabel = str(textData.iloc[0])
			
			key = rowIndex
			xyDataLabel = tuple(self.data[self.numericColumns].loc[rowIndex])
			xLimDelta,yLimDelta = xLim_and_yLim_delta(ax)
			xyText = (xyDataLabel[0]+ xLimDelta*0.02, xyDataLabel[1]+ yLimDelta*0.02)
			annotObject = ax.annotate(s=textLabel,xy=xyDataLabel,xytext= xyText, ha='left', arrowprops=arrow_args)
		
			self.selectionLabels[key] = dict(xy=xyDataLabel, s=textLabel, xytext = xyText)
			self.madeAnnotations[key] = annotObject	
		## redraws added annotations
		if redraw:	
			self.plotter.redraw()

	
	def replotAllAnnotations(self, ax):
		'''
		If user opens a new session, we need to first replot annotation and then enable
		the drag&drop events..
		'''
		
		self.madeAnnotations.clear() 
		for key,annotationProps in self.selectionLabels.items():
			annotObject = ax.annotate(ha='left', arrowprops=arrow_args,**annotationProps)
			self.madeAnnotations[key] = annotObject
		
	def onPressMoveAndRemoveAnnotions(self,event):
		'''
		Depending on which button used by the user, it trigger either moving around (button-1 press)
		or remove the label.
		'''
		if event.inaxes is None:
			return 
		if event.button in [2,3]: ## mac is 2 and windows 3..
			self.remove_clicked_annotation(event)
		elif event.button == 1:
			self.move_annotations_around(event)
	
	def annotate_all_row_in_data(self):
		'''
		'''
		self.addAnnotationFromDf(self.data)
	
			
	def remove_clicked_annotation(self,event):
		'''
		Removes annotations upon click from dicts and figure
		does not redraw canvas
		'''
		self.plotter.castMenu = True
		toDelete = None
		for key,madeAnnotation  in self.madeAnnotations.items():
			if madeAnnotation.contains(event)[0]:
				self.plotter.castMenu = False
				madeAnnotation.remove()
				toDelete = key
				break
		if toDelete is not None:
			del self.selectionLabels[toDelete] 
			del self.madeAnnotations[toDelete] 
			self.eventOverAnnotation = False
			self.plotter.redraw()		
	
	def remove_all_annotations(self):
		'''
		Removes all annotations. Might be called from outside to let 
		the user delete all annotations added
		'''
		
		for madeAnnotation  in self.madeAnnotations.values():
				madeAnnotation.remove()
		self.madeAnnotations.clear()
		self.selectionLabels.clear()

	def find_closest_and_check_event(self,xyEvent,event):
		'''
		'''
		if len(self.selectionLabels) == 0:
			return
		annotationsKeysAndPositions = [(key,annotationDict['xytext']) for key,annotationDict \
		in self.selectionLabels.items()][::-1]
		
		keys, xyPositions = zip(*annotationsKeysAndPositions)
		idxClosest = closest_coord_idx(xyPositions,xyEvent)[0]
		keyClosest = keys[idxClosest]
		annotationClostest = self.madeAnnotations[keyClosest]
		self.eventOverAnnotation = annotationClostest.contains(event)[0]
		
		return annotationClostest,xyPositions,idxClosest,keyClosest

			
	def move_annotations_around(self,event):
		'''
		wrapper to move around labels. We did not use the annotation.draggable(True) option
		because this moves all artists around that are under the mouseevent.
		'''
		if len(self.selectionLabels) == 0 or event.inaxes is None:
			return 
		self.eventOverAnnotation = False	
		
		xyEvent =  (event.xdata,event.ydata)
		
		annotationClostest,xyPositions,idxClosest,keyClosest = \
		self.find_closest_and_check_event(xyEvent,event)	
					
		
		if self.eventOverAnnotation:
			ax = self.ax
			inv = ax.transData.inverted()
			
			#renderer = self.figure.canvas.renderer() 
			xyPositionOfLabelToMove = xyPositions[idxClosest] 
			background = self.plotter.figure.canvas.copy_from_bbox(ax.bbox)
			widthRect, heightRect = self.get_rectangle_size_on_text(ax,annotationClostest.get_text(),inv)
			recetangleToMimicMove = patches.Rectangle(xyPositionOfLabelToMove,width=widthRect,height=heightRect,
													fill=False, linewidth=0.6, edgecolor="darkgrey",
                             						animated = True,linestyle = 'dashed', clip_on = False)
			
			ax.add_patch(recetangleToMimicMove)
			
			self.rectangleMoveEvent = self.plotter.figure.canvas.mpl_connect('motion_notify_event', 
										lambda event:self.move_rectangle_around(event,recetangleToMimicMove,background,inv,ax))
                             									
                             									
			self.releaseLabelEvent = self.plotter.figure.canvas.mpl_connect('button_release_event', 
										lambda event: self.disconnect_label_and_update_annotation(event,
																					recetangleToMimicMove,
																					annotationClostest,
																					keyClosest))
		
	def move_rectangle_around(self,event,rectangle,background,inv, ax):
		'''
		actually moves the rectangle
		'''
		x_s,y_s = event.x, event.y
		x,y= list(inv.transform((x_s,y_s)))
		self.plotter.figure.canvas.restore_region(background)
		rectangle.set_xy((x,y))  
		ax.draw_artist(rectangle)
		self.plotter.figure.canvas.blit(ax.bbox)    
          
	def disconnect_label_and_update_annotation(self,event,rectangle,annotation,keyClosest):
		'''
		Mouse release event. disconnects event handles and updates the annotation dict to
		keep track for export
		'''
		self.plotter.figure.canvas.mpl_disconnect(self.rectangleMoveEvent)
		self.plotter.figure.canvas.mpl_disconnect(self.releaseLabelEvent)
		
		xyRectangle = rectangle.get_xy()
		annotation.set_position(xyRectangle)
		
		rectangle.remove()
		self.plotter.redraw() 
		self.selectionLabels[keyClosest]['xytext'] = xyRectangle

	def get_rectangle_size_on_text(self,ax,text,inv):
		'''
		Returns rectangle to mimic the position
		'''	
		renderer = self.plotter.figure.canvas.get_renderer() 
		fakeText = ax.text(0,0,s=text)
		patch = fakeText.get_window_extent(renderer)
		xy0 = list(inv.transform((patch.x0,patch.y0)))
		xy1 = list(inv.transform((patch.x1,patch.y1)))
		fakeText.remove()
		widthText = xy1[0]-xy0[0]
		heightText = xy1[1]-xy0[1]
		return widthText, heightText
                             
	def update_data(self,data):
		'''
		Updates data to be used. This is needed if the order of the data 
		changed.
		'''
		self.data = data		
