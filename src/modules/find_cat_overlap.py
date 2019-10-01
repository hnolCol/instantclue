


import numpy as np 
import pandas as pd
import itertools


class findCategoricalIntersection(object):

	def __init__(self,categoricalValues,uniqueValues = None, 
					 data = None, numericColumns = [],
					 threshold= 5, sep = ';'):

		self.categoricalValues = categoricalValues
		self.sep = sep
		self.data = data
		self.numericColumns = numericColumns
		self.thres = threshold
		self.saveIndices = dict()

		self.findUniqueValues(uniqueValues)
		self.combs = self.getCombinations(self.uniqValues['UniqueValues'].values.tolist())

	def  findUniqueValues(self,uniqueValues ):
		'''
		'''
		splitData = uniqueValues.astype('str').str.split(self.sep).values
		self.uniqValues = pd.DataFrame(list(set(itertools.chain.from_iterable(splitData))),columns = ['UniqueValues'])

	def getCombinations(self, x):
		'''
		'''
		
		combs = itertools.combinations(x, 2)
		return combs
	
	def generateRegExpres(self,category,sep):
		'''
		'''
		regExp = r''
		regExp = regExp + r'({}{})|(^{}$)|({}{}$)|'.format(category,sep,category,sep,category)
		regExp = regExp[:-1]
		return regExp
		
	def fit(self):
		'''
		'''
		df = pd.DataFrame(columns = ['cat_1','cat_2','n']+ self.numericColumns)
		#print(len(self.combs))
		for n,(cat1,cat2) in enumerate(self.combs):
			
			if cat1 not in self.saveIndices:
				regExp1 = self.generateRegExpres(cat1,self.sep)
				boolIndicator1 = self.categoricalValues.astype(str).str.contains(regExp1)
				firstIdx = [idx for idx in boolIndicator1.index if boolIndicator1.iloc[idx]]
				self.saveIndices[cat1] = firstIdx
			else:
				firstIdx = self.saveIndices[cat1]
			
			if cat2 not in self.saveIndices:
				regExp2 = self.generateRegExpres(cat2,self.sep)
				boolIndicator2 = self.categoricalValues.astype(str).str.contains(regExp2)
				secondIdx = [idx for idx in boolIndicator2.index if boolIndicator2.iloc[idx]]
			else:
				secondIDx = self.saveIndices[cat2] 
						
			intersection = set.intersection(set(firstIdx),
				set(secondIdx))
			nIntersect = len(intersection)
			if  nIntersect > self.thres:
				catInfo = {'cat_1':cat1,'cat_2':cat2,'n':nIntersect}
				print(self.data) 
				print(self.numericColumns)
				if self.data is not None and len(self.numericColumns) > 0:
					df1 = self.data.iloc[intersection,]
					for col in self.numericColumns:
						
						catInfo['mean_{}'.format(col)] = df1[col].mean()
									
				df = df.append(catInfo,ignore_index=True)
				
			if n % 100 == 0: 
				print(n)
		
		return df
			
			
		
		
		
		













