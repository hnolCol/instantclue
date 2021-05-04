#data contains hex colors

from .data import * 
from collections import OrderedDict
import seaborn as sns

class ColorHelper(object):
	def __init__(self):
		self.customColors = OrderedDict()
		self.addDefinedColors()
		self.setCustomPalettes()
		self.addPalette()

	def addPalette(self):
		'''
		'''
		for paletteName, colorList in self.customColors.items():
			sns.palettes.SEABORN_PALETTES[paletteName] = colorList

	def getColorPalettes(self):
		'''
		'''
		return self.preDefinedColors

	def getCustomPallettes(self):
		'''
		'''
		self.customColors

	def addDefinedColors(self):
		'''
		'''
		self.preDefinedColors = OrderedDict([('1-Color-Sequential',oneColorSeq),
											('2-Color-Sequential',twoColorSeq),
											('Diverging',diverging),
											('Qualitative',qualitative)])
	def setCustomPalettes(self):
		'''
		Set Custom Color Palettes 
		'''
		self.customColors['Set7'] = Set7
		self.customColors['Set4'] = Set4
		self.customColors['Set5'] = Set5
		self.customColors['Set6'] = Set6
		self.customColors['Tenenbaums'] = tenenH
		self.customColors['Darjeeling Limited'] = darj
		self.customColors['Moonrise Kingdom'] = moon
		self.customColors['Life Acquatic'] = acqua
		
		#self.save_custom_colors()

	# def add_palette(self,palette):
	# 	'''
	# 	Adds the provided palette and saves it. 
	# 	'''
	# 	nCustom = len(self.customColors)-1
	# 	key = 'Custom {}'.format(nCustom)
	# 	self.customColors[key] = palette
	# 	sns.palettes.SEABORN_PALETTES[key] = palette
	# 	self.preDefinedColors['Qualitative (Custom)'].append(key)
	# 	self.save_custom_colors()
		
	# def add_gradient(self,cmap,type='1-Color-Sequential'):
	# 	'''
	# 	Adds provided gradient (matplotlib cmap object) and
	# 	saves it with a key that indicates what type of gradient it is. 
	# 	'''
	# 	nCustom = len(self.customColors)-1
	# 	key = '{}GRADIENT_Custom {}'.format(type,nCustom)
	# 	cmapName = key.split('GRADIENT_')[-1]
	# 	matplotlib.cm.register_cmap(name = cmapName,cmap = cmap)
	# 	self.customColors[key] = cmap
	# 	self.preDefinedColors[type].append(cmapName)
	# 	self.save_custom_colors()
		
	# def delete_palette(self,key):
	# 	'''
	# 	'''
	# 	for type,palettes in self.preDefinedColors.items():
	# 		if key in palettes:
	# 			idx = palettes.index(key)
	# 			del palettes[idx]
	# 			self.preDefinedColors[type] = palettes
	# 			break
	# 	if type == 'Qualitative (Custom)':
	# 		del self.customColors[key]
	# 	else:
	# 		keyInDict = '{}GRADIENT_{}'.format(type,key)	
	# 		del self.customColors[keyInDict]
			
	# 	self.save_custom_colors()
	# 	return type
					
	# def open_custom_colors(self):
	# 	'''
	# 	Opens the file in which the custom colors are stored. 
	# 	If we cannot find the file. Create a new one.
	# 	'''
		
	# 	if os.path.exists(self.path) == False:
	# 		self.reset_default()
	# 		return
			
	# 	with open(self.path,'rb') as file:
	# 		self.customColors = pickle.load(file)
		
	# 	colorNames = []
	# 	for key,colorList in self.customColors.items():
	# 		if 'GRADIENT' in key:
	# 			keySplit = key.split('GRADIENT_')
	# 			matplotlib.cm.register_cmap(name=keySplit[-1], cmap=colorList)
	# 			self.preDefinedColors[keySplit[0]].append(keySplit[-1])
	# 		else:
	# 			sns.palettes.SEABORN_PALETTES[key] = colorList
	# 			colorNames.append(key)
		
	# 	self.preDefinedColors['Qualitative (Custom)'] = colorNames
	
	# def save_custom_colors(self):
	# 	'''
	# 	Pickles the modified dict containing the colors
	# 	(customColors).
	# 	'''
		
		
	# 	if True:
	# 		if os.path.exists(os.path.join(path_file,'data')) == False:
	# 			os.mkdir(os.path.join(path_file,'data'))
	# 			os.mkdir(os.path.join(path_file,'data','colors'))
	# 			print('created folder')
	# 		with open(self.path,'wb') as file:
	# 			pickle.dump(self.customColors, file)
	# 	else:
	# 		try:
	# 			tk.messagebox.showinfo('Error ..','Color package file not found or no permission?')
	# 		except:
	# 			pass
