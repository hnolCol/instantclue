"""
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
import scipy
from scipy.signal import detrend
from collections import OrderedDict

'''
============ PLEASE NOTE ===============
*The main part of this code is a copy of pyvttbl that was designed for python2 and 
*had its own dataframe structure. Here it is adjusted to work with pandas dataframes and
*python3. 
*### FOR ANY CROSS REFERENCES AND LICENCE CHECK OUT:
*https://pypi.python.org/pypi/pyvttbl
'''


def epsGG(y, df1):
    """
    (docstring is adapted from Trujillo-Ortiz (2006); see references)
    
    The Greenhouse-Geisser epsilon value measures by how much the
    sphericity assumption is violated. Epsilon is then used to adjust
    for the potential bias in the F statistic. Epsilon can be 1, which
    means that the sphericity assumption is met perfectly. An epsilon
    smaller than 1 means that the sphericity assumption is violated.
    The further it deviates from 1, the worse the violation; it can be
    as low as epsilon = 1/(k - 1), which produces the lower bound of
    epsilon (the worst case scenario). The worst case scenario depends
    on k, the number of levels in the repeated measure factor. In real
    life epsilon is rarely exactly 1. If it is not much smaller than 1,
    then we feel comfortable with the results of repeated measure ANOVA. 
    The Greenhouse-Geisser epsilon is derived from the variance-
    covariance matrix of the data. For its evaluation we need to first
    calculate the variance-covariance matrix of the variables (S). The
    diagonal entries are the variances and the off diagonal entries are
    the covariances. From this variance-covariance matrix, the epsilon
    statistic can be estimated. Also we need the mean of the entries on
    the main diagonal of S, the mean of all entries, the mean of all
    entries in row i of S, and the individual entries in the variance-
    covariance matrix. There are three important values of epsilon. It
    can be 1 when the sphericity is met perfectly. This epsilon
    procedure was proposed by Greenhouse and Geisser (1959). Greenhouse-
    Geisser's epsilon is calculated using the Satterthwaite
    approximation. See Glaser (2003.)
    
      Syntax: function epsGG(y,df1)
    
      Inputs:
         y   = Input matrix can be a data matrix
               (size n-data x k-treatments)
         df1 = degrees of freedom of treatment
         
      Output:
         Greenhouse-Geisser epsilon value.
    
     $$We suggest you could take-a-look to the PDF document ''This Week's 
       Citation Classics'' CCNumber 28, July 12, 1982, web-page
       [http://garfield.library.upenn.edu/classics1982/A1982NW45700001.pdf]$$
    
    Example 2 of Maxwell and Delaney (p.497). This is a repeated measures
    example with two within and a subject effect. We have one dependent
    variable:reaction time, two independent variables: visual stimuli
    are tilted at 0, 4, and 8 degrees; with noise absent or present.
    Each subject responded to 3 tilt and 2 noise given 6 trials. Data are,
    
                          0           4           8                  
                     -----------------------------------
            Subject    A     P     A     P     A     P
            --------------------------------------------
               1      420   480   420   600   480   780
               2      420   360   480   480   480   600
               3      480   660   480   780   540   780
               4      420   480   540   780   540   900
               5      540   480   660   660   540   720
               6      360   360   420   480   360   540
               7      480   540   480   720   600   840
               8      480   540   600   720   660   900
               9      540   480   600   720   540   780
              10      480   540   420   660   540   780
            --------------------------------------------
    
    The three measurements of reaction time were averaging across noise 
    ausent/present. Given,
    
                             Tilt
                      -----------------
            Subject     0     4     8    
            ---------------------------
               1       450   510   630
               2       390   480   540
               3       570   630   660
               4       450   660   720
               5       510   660   630
               6       360   450   450
               7       510   600   720
               8       510   660   780
               9       510   660   660
              10       510   540   660
            ---------------------------
    
    We need to estimate the Greenhouse-Geisser epsilon associated with
    the angle of rotation of the stimuli. 
    
    Reference:
      Glaser, D.E. (2003). Variance Components. In R.S.J. Frackowiak, K.J.
          Friston, C. Firth, R. Dolan, C.J., Price, S. Zeki, J. Ashburner,
          & W.D. Penny, (Eds.), Human Brain Function. Academic Press, 2nd.
          edition. [http://www.fil.ion.ucl.ac.uk/spm/doc/books/hbf2/]
      Greenhouse, S.W. and Geisser, S. (1959), On methods in the analysis
          of profile data. Psychometrika, 24:95-112. 
      Maxwell, S.E. and Delaney, H.D. (1990), Designing Experiments and 
          Analyzing Data: A model comparison perspective. Pacific Grove,
          CA: Brooks/Cole.
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). epsGG:Greenhouse-Geisser epsilon. A MATLAB
          file. [WWW document]. URL
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
    
    V = np.cov(y) # sample covariance
    return np.trace(V)**2 / (df1*np.trace(np.dot(V.T,V)))      

def epsHF(y, df1):
    """
    This is ported from a Matlab function written by Trujillo-Ortiz et
    al. 2006 (see references) with an important modification. If the
    calculated epsilon values is greater than 1, it returns 1. 
    
    The Huynh-Feldt epsilon its a correction of the Greenhouse-Geisser
    epsilon. This due that the Greenhouse-Geisser epsilon tends to
    underestimate epsilon when epsilon is greater than 0.70 (Stevens,
    1990). An estimated epsilon = 0.96 may be actually 1. Huynh-Feldt
    correction is less conservative. The Huynh-Feldt epsilon is
    calculated from the Greenhouse-Geisser epsilon. As the Greenhouse-
    Geisser epsilon, Huynh-Feldt epsilon measures how much the
    sphericity assumption or compound symmetry is violated. The idea of
    both corrections its analogous to pooled vs. unpooled variance
    Student's t-test: if we have to estimate more things because
    variances/covariances are not equal, then we lose some degrees of
    freedom and P-value increases. These epsilons should be 1.0 if
    sphericity holds. If not sphericity assumption appears violated.
    We must to have in mind that the greater the number of repeated
    measures, the greater the likelihood of violating assumptions of
    sphericity and normality (Keselman et al, 1996) . Therefore, we need
    to have the most conservative F values. These are obtained by
    setting epsilon to its lower bound, which represents the maximum
    violation of these assumptions. When a significant result is
    obtained, it is assumed to be robust. However, since this test may
    be overly conservative, Greenhouse and Geisser (1958, 1959)
    recommend that when the lower-bound epsilon gives a nonsignificant
    result, it should be followed by an approximate test (based on a
    sample estimate of epsilon).

      Syntax: function epsHF(y,df1)
    
      Inputs:
         y   = Input matrix can be a data matrix
               (size n-data x k-treatments)
         df1 = degrees of freedom of treatment
         
      Output:
         Huynh-Feldt epsilon value.

    See docstring for epsGG() for information on formatting X.
    
    Reference:
      Geisser, S, and Greenhouse, S.W. (1958), An extension of Box's
          results on the use of the F distribution in multivariate
          analysis. Annals of Mathematical Statistics, 29:885-891.
      Greenhouse, S.W. and Geisser, S. (1959), On methods in the
          analysis of profile data. Psychometrika, 24:95-112. 
      Huynh, M. and Feldt, L.S. (1970), Conditions under which mean
          square rate in repeated measures designs have exact-F
          distributions. Journal of the American Statistical
          Association, 65:1982-1989 
      Keselman, J.C, Lix, L.M. and Keselman, H.J. (1996), The analysis
          of repeated measurements: a quantitative research synthesis.
          British Journal of Mathematical and Statistical Psychology,
          49:275-298.
      Maxwell, S.E. and Delaney, H.D. (1990), Designing Experiments
          and Analyzing Data: A model comparison perspective. Pacific
          Grove, CA: Brooks/Cole.
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). epsGG:Greenhouse-Geisser epsilon. A
          MATLAB file. [WWW document].
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
    
    k,n = np.shape(y)      # number of treatments
    eGG = epsGG(y, df1) # Greenhouse-Geisser epsilon

    N = n*(k-1.)*eGG-2.
    D = (k-1.)*((n-1.)-(k-1.)*eGG)
    eHF = N/D                 # Huynh-Feldt epsilon estimation

    if   eHF < eGG : return eGG
    elif eHF > 1.  : return 1.
    else           : return eHF

def epsLB(y, df1):
    """
    This is ported from a Matlab function written by Trujillo-Ortiz et
    al. 2006. See references.
    
    EPBG Box's conservative epsilon.
    The Box's conservative epsilon value (Box, 1954), measures by how
    much the sphericity assumption is violated. Epsilon is then used to
    adjust for the potential bias in the F statistic. Epsilon can be 1,
    which means that the sphericity assumption is met perfectly. An
    epsilon smaller than 1 means that the sphericity assumption is
    violated. The further it deviates from 1, the worse the violation;
    it can be as low as epsilon = 1/(k - 1), which produces the lower
    bound of epsilon (the worst case scenario). The worst case scenario
    depends on k, the number of levels in the repeated measure factor.
    In real life epsilon is rarely exactly 1. If it is not much smaller
    than 1, then we feel comfortable with the results of repeated
    measure ANOVA. The Box's conservative epsilon is derived from the
    lower bound of epsilon, 1/(k - 1). Box's conservative epsilon is no
    longer widely used. Instead, the Greenhouse-Geisser's epsilon
    represents its maximum-likelihood estimate.
    
      Syntax: function epsLB(y,df1)
    
      Inputs:
         y   = Input matrix can be a data matrix
               (size n-data x k-treatments)
         df1 = degrees of freedom of treatment
         
      Output:
         Box's conservative epsilon value.

    See docstring for epsGG() for information on formatting X.
    
    Reference:
      Box, G.E.P. (1954), Some theorems on quadratic forms applied in
          the study of analysis of variance problems, II. Effects of
          inequality of variance and of correlation between errors in
          the two-way classification. Annals of Mathematical Statistics.
          25:484-498. 
      Trujillo-Ortiz, A., R. Hernandez-Walls, A. Castro-Perez and K.
          Barba-Rojo. (2006). epsGG:Greenhouse-Geisser epsilon. A MATLAB
          file. [WWW document]. 
          http://www.mathworks.com/matlabcentral/fileexchange
          /loadFile.do?objectId=12839
    """
    if df1 == 1. : return 1.
        
    k = np.shape(y)[0]  # number of treatments
    box = 1./(k-1.) # Box's conservative epsilon estimation

    if box*df1 < 1. : box = 1. / df1
    
    return box
    
calculationTypes = OrderedDict([('Sphericity Assumed',''),
									   ('Greenhouse-Geisser','_gg'),
									   ('Huynh-Feldt','_hf'),
									   ('Box','_lb')])
		
columnHeaderDictKey = OrderedDict([('Source',''),('Correction','type'),
										  ('Type III SS','ss'),('eps','eps'),
										  ('df','df'),('MS','mss'),('F-value','F'),
										  ('p-value','p'),('et2_G','eta'),
										  ('Observations','obs'),('SE','se'),
										  ('95% CI','ci'),('lambda','lambda'),
										  ('Observed Power','power')])    
    
    
class Anova(object):
	
	def __init__(self, dataframe, dependentVariable, wFactors = [], bFactors = [],
					subjectColumn = 'Subject', measure = '', alpha = 0.05):		
		'''
		Paramater
		==========
		'''
		
		
		factors = wFactors + bFactors
		self.subj = subjectColumn
		## getting unique levels for each factor in dict as well as 
		##their size in a list the order is as the one in factors
		self.uniqueLevels, self.numbUniqueLevels = self.get_unique_categorical_values(dataframe,factors)
		self.results = OrderedDict()
		
		self.df_pivot = pd.pivot_table(
									data =  dataframe, 
									index = [subjectColumn],
									values = dependentVariable,
									columns = factors,
									aggfunc = np.mean)
		self.df_pivot_array = self.df_pivot.values
		self.df_pivot_array[np.isnan(self.df_pivot_array)] = dataframe[dependentVariable].mean() 
		
		lenWFactors, lenBFactors = len(wFactors),len(bFactors)
		
		if lenWFactors != 0 and lenBFactors == 0:
			self.withinFactorsDesign(wFactors, dataframe)
			self.finalResult, self.title = self.pack_results_in_df_within(dependentVariable,wFactors)
			
		if lenWFactors == 0 and lenBFactors != 0:
			self.betweenFactorsDesign(bFactors,dataframe)
			self.finalResult, self.title = self.pack_results_in_df_between(dependentVariable, bFactors)
			
		if lenWFactors != 0 and lenBFactors != 0:
			self.mixedFactorsDesign(dependentVariable,wFactors,bFactors,dataframe)
			self.finalResult, self.title = self.pack_results_in_df_mixed(dependentVariable, wFactors,bFactors)
			
	def getResults(self):
		""
		if hasattr(self,"finalResult"):
			return self.finalResult


	def betweenFactorsDesign(self,bFactors,dataframe):
		'''
		'''
		Nr,Nn = self.df_pivot.shape # Nr - number of replicates 
									 #(subjects), Nn number of treatments
		D = self.numbUniqueLevels
		Nf = len(D)  # Number of facotrs
		Nd = np.prod(D) # Total number of conditions
		Ne = 2**Nf - 1 # number of effects
		
		contrasts, meanEffects = {}, {} # mean Effects might be bad naming?
		for f in range(1,Nf+1):
			# result dict
			r = {}
			## create main effect/interaction component contrasts
			contrasts[(f,1)] = np.ones((D[f-1],1))
			contrasts[(f,2)] = detrend(np.eye(D[f-1]), type = 'constant')
			
			## create main effect/interaction components for means
			meanEffects[(f,1)] = np.ones((D[f-1],1))/D[f-1]
			meanEffects[(f,2)] = np.eye(D[f-1])
			
		for effect in range(1,Ne+1):
			cw = self.num2binvec(effect,Nf)
			# this steps gets the factor names
			# example efs = ['AGE'] or efs = ['AGE','CONDITION']
			efs = np.asarray(bFactors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
			# get full contrasts
			r = {}
			c = contrasts[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				c = np.kron(c,contrasts[(f,cw[Nf-f])])
			Nc = np.shape(c)[1] # Number of condition in effect
			No = Nd/Nc*1. #Number of observations per condition in effect
			y = np.dot(self.df_pivot_array,c) 
			nc = np.shape(y)[1]
			
			cy = meanEffects[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				cy = np.kron(cy,meanEffects[(f,cw[Nf-f])])
			
			r['y2'] = np.mean(np.dot(self.df_pivot_array,cy),0)
			
			b = np.mean(y,0)
			r['df'] = float(self.matrix_rank(c))
			r['ss'] = np.sum(y*b.T)*Nc
			r['mss'] = r['ss']/r['df']
						
			self.results[tuple(efs)] = r
		
		ssTotal = np.sum((self.df_pivot_array-np.mean(self.df_pivot_array))**2)
		ssError = ssTotal
		dfe = len(dataframe.index)-1
		
		for i in range(1,len(bFactors)+1):
			for efs in self.unique_combinations(bFactors,i):
				ssError -= self.results[tuple(efs)]['ss']
				dfe -= self.results[tuple(efs)]['df']
		
		for i in range(1,len(bFactors)+1):
			for efs in self.unique_combinations(bFactors,i):
				
				r = self.results[tuple(efs)]
				
				r['sse'] = ssError
				r['dfe'] = dfe
				r['mse'] = ssError/dfe
				r['F'] = r['mss']/r['mse']
				r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
				
				## generalized eta effect size
				r['eta'] = r['ss']/(r['ss']+r['sse'])
				
				# observations per cell
				r['obs'] = dataframe[self.subj].unique().size
				r['obs'] /= np.prod([self.uniqueLevels[f].size for f in efs]) 
				# Loftus and Masson standard errors
				r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
				r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
				r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']
				
				p_eta2 = r['ss']/(r['ss']+r['sse'])
				r['lambda'] = (p_eta2/(1-p_eta2))*r['obs']
				r['power'] = self.observed_power( r['df'], r['dfe'], r['lambda'] )
				
				self.results[tuple(efs)] = r
				
					
		
	def withinFactorsDesign(self,wFactors,dataframe):
		'''
		'''
		
		Nr,Nn = self.df_pivot.shape # Nr - number of replicates 
									 #(subjects), Nn number of treatments
		D = self.numbUniqueLevels
		Nf = len(D)  # Number of facotrs
		Nd = np.prod(D) # Total number of conditions
		Ne = 2**Nf - 1 # number of effects		
		
		contrasts, meanEffects = {}, {} # mean Effects might be bad naming?
		for f in range(1,Nf+1):
			## create main effect/interaction component contrasts
			contrasts[(f,1)] = np.ones((D[f-1],1))
			contrasts[(f,2)] = detrend(np.eye(D[f-1]), type = 'constant')
			
			## create main effect/interaction components for means
			meanEffects[(f,1)] = np.ones((D[f-1],1))/D[f-1]
			meanEffects[(f,2)] = np.eye(D[f-1])		
		
		dfeSum = 0
		for i in range(1,len(wFactors)+1):
			for efs in self.unique_combinations(wFactors,i):
				#intermediate result dict r
				r = {}
				r['df'] = np.prod([self.uniqueLevels[f].size-1 for f in wFactors if f in efs])
				r['dfe'] = float(r['df']*(Nr-1))
				self.results[tuple(efs)] = r
				
		for effect in range(1,Ne+1):
			
			cw = self.num2binvec(effect,Nf)
			# this steps gets the factor names
			# example efs = ['AGE'] or efs = ['AGE','CONDITION']
			efs = np.asarray(wFactors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
			r=self.results[tuple(efs)]
			# get full contrasts
			c = contrasts[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				c = np.kron(c,contrasts[(f,cw[Nf-f])])
			Nc = np.shape(c)[1] # Number of condition in effect
			No = Nd/Nc*1. #Number of observations per condition in effect
			y = np.dot(self.df_pivot_array,c) 
			nc = np.shape(y)[1]		
			
			cy = meanEffects[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				cy = np.kron(cy,meanEffects[(f,cw[Nf-f])])
			
			r['y2'] = np.mean(np.dot(self.df_pivot_array,cy),0)
			r['eps_gg'] = epsGG(y,r['df'])
			r['eps_hf'] = epsHF(y,r['df'])
			r['eps_lb'] = epsLB(y,r['df'])
			
			# calculate ss, sse, mse, mss, F, p, and standard errors

			b = np.mean(y,0)
			# Sphericity assumed
			
			r['ss']   =  np.sum(y*b.T)
			r['mse']  = (np.sum(np.diag(np.dot(y.T,y)))-r['ss'])/r['dfe']
			r['sse']  =  r['dfe']*r['mse']
			r['ss']  /=  No
			r['mss'] =  r['ss']/r['df']
			r['sse'] /=  No
			r['mse'] =  r['sse']/r['dfe']
			r['F'] =  r['mss']/r['mse']
			r['p'] =  scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
			# calculate observations per cell
			r['obs'] =  Nr*No
			# calculate Loftus and Masson standard errors
			r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
			r['se'] = np.sqrt(r['mse']/r['obs'])*r['critT']/1.96
			r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']
			
			p_eta2 = r['ss']/(r['ss']+r['sse'])
			r['lambda'] = (p_eta2/(1-p_eta2))*r['obs']
			r['power'] = self.observed_power(r['df'], r['dfe'], r['lambda'])
			
			# Greenhouse-Geisser, Huynh-Feldt, Lower-Bound
			for x in ['_gg','_hf','_lb']:
				r['df%s'%x]  = r['df']*r['eps%s'%x]
				r['dfe%s'%x] = r['dfe']*r['eps%s'%x]
				r['mss%s'%x] = r['ss']/r['df%s'%x]
				r['mse%s'%x] = r['sse']/r['dfe%s'%x]
				r['F%s'%x] = r['mss%s'%x]/r['mse%s'%x]
				r['p%s'%x] = scipy.stats.f(r['df%s'%x],r['dfe%s'%x]).sf(r['F%s'%x])
				r['obs%s'%x] = Nr*No
				r['critT%s'%x] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
				r['se%s'%x] =np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]/1.96
				r['ci%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]
				# calculate non-centrality and observed power#
				r['lambda%s'%x]=r['lambda']
				r['power%s'%x]=self.observed_power( r['df'], r['dfe'], r['lambda'] ,eps=r['eps%s'%x])
			self.results[tuple(efs)] = r
		subMeans = np.mean(self.df_pivot_array, axis= 1) 
		ssSubject = np.sum((subMeans-np.mean(self.df_pivot_array))**2)
		ssSubject *= (np.prod([self.uniqueLevels[f].size for f in wFactors])*1.) #ensure float
		ssErrorTotal = np.sum([r['sse'] for r in self.results.values()])
		for efs, r in self.results.items():
			r['eta'] = r['ss']/ (ssSubject + ssErrorTotal)
			self.results[tuple(efs)] = r
 		
 		 
	 		
	def mixedFactorsDesign(self,dependentVariable,wFactors,bFactors,dataframe):
		'''
		'''
		factors = wFactors+bFactors
		Nr,Nn = self.df_pivot.shape # Nr - number of replicates 
									 #(subjects), Nn number of treatments
		D = self.numbUniqueLevels
		Nf = len(D)  # Number of factors
		Nd = np.prod(D) # Total number of conditions
		Ne = 2**Nf - 1 # number of effects		
		
		contrasts, meanEffects = {}, {} # mean Effects might be bad naming?
		for f in range(1,Nf+1):

			## create main effect/interaction component contrasts
			contrasts[(f,1)] = np.ones((D[f-1],1))
			contrasts[(f,2)] = detrend(np.eye(D[f-1]), type = 'constant')
			
			## create main effect/interaction components for means
			meanEffects[(f,1)] = np.ones((D[f-1],1))/D[f-1]
			meanEffects[(f,2)] = np.eye(D[f-1])				
		
		for effect in range(1,Ne+1):
			
			cw = self.num2binvec(effect,Nf)
			# this steps gets the factor names
			# example efs = ['AGE'] or efs = ['AGE','CONDITION']
			efs = np.asarray(factors)[Nf-1-np.where(np.asarray(cw)==2.)[0][::-1]]
			r= {}
			# get full contrasts
			c = contrasts[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				c = np.kron(c,contrasts[(f,cw[Nf-f])])
			Nc = np.shape(c)[1] # Number of condition in effect
			No = Nd/Nc*1. #Number of observations per condition in effect
			y = np.dot(self.df_pivot_array,c) 
			nc = np.shape(y)[1]	
			
			cy = meanEffects[(1,cw[Nf-1])]
			for f in range(2,Nf+1):
				cy = np.kron(cy,meanEffects[(f,cw[Nf-f])])
			
			r['y2'] = np.mean(np.dot(self.df_pivot_array,cy),0)
			# df for effect
			r['df'] =  np.prod([self.uniqueLevels[f].size-1 for f in efs])
			
			r['eps_gg'] = epsGG(y,r['df'])
			r['eps_hf'] = epsHF(y,r['df'])
			r['eps_lb'] = epsLB(y,r['df'])	
			
			b = np.mean(y,0)
					
			# Sphericity assumed
			
			r['ss'] = np.sum(y*b.T)	
			r['ss'] /= No/(np.prod([self.uniqueLevels[f].size for f in bFactors])*1.)	
			r['mss'] = r['ss']/r['df']
			self.results[tuple(efs)] = r
 			
		ssTotal = np.sum((self.df_pivot_array - np.mean(self.df_pivot_array))**2)
		subMeans = dataframe.groupby([self.subj])[dependentVariable].mean().values
		ssBSub = np.sum((subMeans-np.mean(self.df_pivot_array))**2)
		ssBSub *= (np.prod([self.uniqueLevels[f].size for f in wFactors])*1.)
		
		df_b = np.prod([self.uniqueLevels[f].size for f in bFactors])
		df_b *= Nr/np.prod([self.uniqueLevels[f].size for f in bFactors])
		df_b -= 1
		
		dfe_b = np.prod([self.uniqueLevels[f].size for f in bFactors])
		dfe_b *= (Nr/np.prod([self.uniqueLevels[f].size for f in bFactors])-1)
		
		dfe_sum = 0	
		dfe_sum += dfe_b
		
		sseB = ssBSub
		self.befs = [] # list of between subjects effects
		for i in range(1,len(bFactors)+1):
			for efs in self.unique_combinations(bFactors,i):
				sseB -= self.results[tuple(efs)]['ss']
				self.befs.append(efs)
		
		mseB = sseB/dfe_b
		
		
		self.results[(self.subj,)] = {'ss'  : ssBSub,
									  'sse' : sseB,
									  'mse' : mseB,
									  'df'  : df_b,
									  'dfe' : dfe_b}
		self.results[('TOTAL',)] = {'ss' : ssTotal,
									'df' : Nr/np.prod([self.uniqueLevels[f].size for f in bFactors])*Nd-1}
		self.results[('WITHIN',)] = {'ss' : ssTotal-ssBSub,
									 'df' : self.results[('TOTAL',)]['df'] - df_b}
 					
		ssErrorTotal = 0
		self.wefs = []
		for i in range(1, len(wFactors)+1):
			for efs in self.unique_combinations(wFactors,i):
				
				self.wefs.append(efs)
				efs+=[self.subj]
				r = {}
				tmp = pd.pivot_table(dataframe, values = dependentVariable, columns = efs[:-1], index= [self.subj])
				r['ss'] = np.sum((tmp-np.mean(self.df_pivot_array))**2)
				r['ss'] *= np.prod([self.uniqueLevels[f].size for f in wFactors if f not in efs]) 
				r['ss'] = r['ss'].sum()
				for j in range(1, len(efs+bFactors)+1):
					for efs2 in self.unique_combinations(efs+bFactors, j):
					
						if efs2 not in self.befs and efs2!=efs:
							if self.subj in efs2 and len(set(efs2).intersection(set(bFactors)))>0:
								pass
							else:
								r['ss'] -= self.results[tuple(efs2)]['ss']


				
				ssErrorTotal += r['ss']
				r['df'] = np.prod([self.uniqueLevels[f].size for f in bFactors])
				r['df'] *= np.prod([self.uniqueLevels[f].size-1 for f in efs if f in wFactors])
				r['df'] *= (Nr/np.prod([self.uniqueLevels[f].size for f in bFactors])-1)
				dfe_sum += r['df']
				
				r['mss'] = r['ss']/r['df']
				
				self.results[tuple(efs)] = r
		ssErrorTotal += mseB*dfe_b
 		
 		
		
		for i in range(1, len(bFactors)+1):
			##calculate mse, dfe, sse F, p and standard errors
			##between subjects effects
			for efs in self.unique_combinations(bFactors,i):
				r = self.results[tuple(efs)]
				r['sse'] = mseB*dfe_b
				r['dfe'] = self.results[(self.subj,)]['dfe']
				r['mse'] = r['sse']/r['dfe']
				r['F'] = r['mss']/r['mse']
				r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
				
				r['eta'] = r['ss']/(r['ss']+ssErrorTotal)
				
				r['obs'] = dataframe[self.subj].unique().size
				r['obs'] /= np.prod([self.uniqueLevels[f].size for f in efs])
				
				
				# calculate Loftus and Masson standard errors
				r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2))
				r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']
				r['se'] = r['ci']/1.96
				
				p_eta2 = r['ss']/(r['ss']+r['sse'])
				r['lambda'] = (p_eta2/(1-p_eta2))*r['obs']
				r['power'] = self.observed_power(r['df'],r['dfe'],r['lambda'])
								
				self.results[tuple(efs)] = r

		for i in range(1, len(factors)+1):
			for efs in self.unique_combinations(factors,i):
				if efs not in self.befs:
					r = self.results[tuple(efs)]
					r2 = self.results[tuple([f for f in efs if f not in bFactors] + [self.subj])]
					
					r['dfe'] = r2['df']
					
					r['sse'] = r2['ss']
					r['mse'] = r2['mss']
					r['F'] = r['mss']/r['mse']
					r['p'] = scipy.stats.f(r['df'],r['dfe']).sf(r['F'])
					
					r['eta'] = r['ss']/(r['ss'] + ssErrorTotal)
					
					r['obs'] = Nr/np.prod([self.uniqueLevels[f].size for f in bFactors])
					r['obs'] *= np.prod([self.uniqueLevels[f].size for f in factors])
					r['obs'] /= np.prod([self.uniqueLevels[f].size for f in efs])
					
					
					# calculate Loftus and Masson standard errors
					r['critT'] = abs(scipy.stats.t(r['dfe']).ppf(.05/2))
					r['ci'] = np.sqrt(r['mse']/r['obs'])*r['critT']
					r['se'] = r['ci']/1.96	
					p_eta2 = r['ss']/(r['ss']+r['sse'])
					r['lambda'] = (p_eta2/(1-p_eta2))*r['obs']
					r['power'] = self.observed_power(r['df'], r['dfe'], r['lambda'])
			
					# Greenhouse-Geisser, Huynh-Feldt, Lower-Bound
					for x in ['_gg','_hf','_lb']:
						r['df%s'%x]  = r['df']*r['eps%s'%x]
						r['dfe%s'%x] = r['dfe']*r['eps%s'%x]
						r['mss%s'%x] = r['ss']/r['df%s'%x]
						r['mse%s'%x] = r['sse']/r['dfe%s'%x]
						r['F%s'%x] = r['mss%s'%x]/r['mse%s'%x]
						r['p%s'%x] = scipy.stats.f(r['df%s'%x],r['dfe%s'%x]).sf(r['F%s'%x])
						r['obs%s'%x] = r['obs']
						r['critT%s'%x] = abs(scipy.stats.t(r['dfe']).ppf(.05/2.))
						r['se%s'%x] =np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]/1.96
						r['ci%s'%x] = np.sqrt(r['mse']/r['obs%s'%x])*r['critT%s'%x]
						# calculate non-centrality and observed power#
						r['lambda%s'%x]=r['lambda']
						r['power%s'%x]=self.observed_power( r['df'], r['dfe'], r['lambda'] ,eps=r['eps%s'%x])
					self.results[tuple(efs)] = r					
									
 			
 			
	def pack_results_in_df_within(self,dependentVariable,wFactors):
		'''
		''' 

					
		title = 'TESTS OF WITHIN SUBJECTS EFFECTS - Measure of {}'.format(dependentVariable)
		
		collectionDict = OrderedDict()
		
		for columnHeader in columnHeaderDictKey.keys():
			collectionDict[columnHeader] = []
			
		
		for i in range(1,len(wFactors)+1):
			for efs in self.unique_combinations(wFactors,i):
				r = self.results[tuple(efs)]
				source =' *'.join(efs)
								
				for columnHeader, dictKey in columnHeaderDictKey.items():
					
					for type, suffix in calculationTypes.items():	
						if dictKey == '':
							collectionDict[columnHeader].append(source)
						elif dictKey == 'type':
							collectionDict[columnHeader].append(type)							
						elif dictKey == 'ss':
							collectionDict[columnHeader].append(r[dictKey])
						else:
							valueKey = dictKey+suffix
							if valueKey in r:
								collectionDict[columnHeader].append(r[valueKey])
							else:
								collectionDict[columnHeader].append('-')
								
				errorSource = 'Error({})'.format(source)
				
				for columnHeader, dictKey in columnHeaderDictKey.items():
					for type, suffix in calculationTypes.items():
						if dictKey == '':
							collectionDict[columnHeader].append(errorSource)
						elif dictKey == 'type':
							collectionDict[columnHeader].append(type)
						elif dictKey in ['ss','eps','df','mss']:
							if dictKey == 'ss':
								valueKey = '{}e'.format(dictKey)
							elif dictKey == 'eps':
								valueKey = '{}{}'.format(dictKey,suffix)
							elif dictKey == 'mss':
								valueKey = '{}{}'.format('mse',suffix)
							else:
								valueKey = '{}e{}'.format(dictKey,suffix)
							if valueKey in r:
								collectionDict[columnHeader].append(r[valueKey])
							else:
								collectionDict[columnHeader].append('-')
						else:
							collectionDict[columnHeader].append('')
				
				
		
		resultDataFrame = pd.DataFrame.from_dict(collectionDict)
		return resultDataFrame, title
						
						
				
	def pack_results_in_df_between(self,dependentVariable,bFactors):
		'''
		'''
		title = 'TESTS OF BETWEEN SUBJECTS EFFECTS - Measure of {}'.format(dependentVariable)
		collectionDict = OrderedDict()
		for columnHeader in columnHeaderDictKey.keys():
			collectionDict[columnHeader] = []
					
		for i in range(1,len(bFactors)+1):
			for efs in self.unique_combinations(bFactors,i):
				r = self.results[tuple(efs)]
				source ='*'.join(efs)
								
				for columnHeader, dictKey in columnHeaderDictKey.items():		
						if dictKey == '':
							collectionDict[columnHeader].append(source)
						else:
							if dictKey in r:
								collectionDict[columnHeader].append(r[dictKey])
							else:
								collectionDict[columnHeader].append('-')				
					
		for columnHeader, dictKey in columnHeaderDictKey.items():
				if columnHeader == 'Source':
					collectionDict[columnHeader].append('Error')
			
				elif dictKey in ['df','mss','ss']:
					if dictKey == 'df':
						collectionDict[columnHeader].append(r['dfe'])
					elif dictKey == 'ss':
						collectionDict[columnHeader].append(r['sse'])
					elif dictKey == 'mss':
						collectionDict[columnHeader].append(r['mse'])
				else:
					collectionDict[columnHeader].append('')
		del collectionDict['eps']
		del collectionDict['Correction']
		resultDataFrame = pd.DataFrame.from_dict(collectionDict)
		return resultDataFrame, title
					

										  
	def pack_results_in_df_mixed(self,dependentVariable,wFactors,bFactors):
		'''

			
		Parameters 
		============
			-dependent Variable - 
				column name of the measurement
			- wFactors
				within-subject effects
			- bFactors
				between-subject effects	
		Output
		============
			two panda dataframes: 
				- between subject results
				- within subject results		
		'''
		
		titleBetween = 'TESTS OF BETWEEN SUBJECTS EFFECTS - Measure of {}'.format(dependentVariable)
		collectionDict = OrderedDict()
		for columnHeader in columnHeaderDictKey.keys():
			collectionDict[columnHeader] = []
		
		for columnHeader in columnHeaderDictKey.keys():
			if columnHeader == 'Source':
				collectionDict[columnHeader].append('Between subjects')
			elif columnHeader == 'Type III SS':
				collectionDict[columnHeader].append(self.results[(self.subj,)]['ss'])
			elif columnHeader == 'df':
				collectionDict[columnHeader].append(self.results[(self.subj,)]['df'])
			else:
				collectionDict[columnHeader].append('-')
						
		for i in range(1,len(bFactors)+1):
			for efs in self.unique_combinations(bFactors,i):
				r = self.results[tuple(efs)]
				source =' *'.join(efs)
				for columnHeader, dictKey in columnHeaderDictKey.items():		
						if dictKey == '':
							collectionDict[columnHeader].append(source)
						else:
							if dictKey in r:
								collectionDict[columnHeader].append(r[dictKey])
							else:
								collectionDict[columnHeader].append('-')
								
		for columnHeader in columnHeaderDictKey.keys():
			if columnHeader == 'Source':
				collectionDict[columnHeader].append('Error')
			elif columnHeader == 'Type III SS':
				collectionDict[columnHeader].append(self.results[(self.subj,)]['sse'])
			elif columnHeader == 'df':
				collectionDict[columnHeader].append(self.results[(self.subj,)]['dfe'])
			elif columnHeader == 'MS':
				collectionDict[columnHeader].append(self.results[(self.subj,)]['mse'])				
			else:
				collectionDict[columnHeader].append('-')
														
		del collectionDict['eps']
		del collectionDict['Correction']
					
		betweenDataFrame = pd.DataFrame.from_dict(collectionDict)	
		del collectionDict
		
		titleWithin = 'TESTS OF WITHIN SUBJECTS EFFECTS - Measure of {}'.format(dependentVariable)
		collectionDict = OrderedDict()
		#reset dict
		for columnHeader in columnHeaderDictKey.keys():
			collectionDict[columnHeader] = []
			
		factors = wFactors + bFactors	
		defs = []	
		for i in range(1,len(wFactors)+1):
			for efs in self.unique_combinations(wFactors,i):
				r = self.results[tuple(efs)]
				source =' * '.join(efs)
				defs.append(efs)
								
				for columnHeader, dictKey in columnHeaderDictKey.items():
					
					for type, suffix in calculationTypes.items():	
						if dictKey == '':
							collectionDict[columnHeader].append(source)
						elif dictKey == 'type':
							collectionDict[columnHeader].append(type)							
						elif dictKey == 'ss':
							collectionDict[columnHeader].append(r[dictKey])
							
						else:
							valueKey = dictKey+suffix
							if valueKey in r:
								collectionDict[columnHeader].append(r[valueKey])
								
							else:
								collectionDict[columnHeader].append('-')
				
				
				for i in range(1,len(factors)+1):
					for efs2 in self.unique_combinations(factors,i):
						if efs2 not in self.befs and efs2 not in defs and efs2 not in self.wefs \
						and len(set(efs2).difference(set(efs+bFactors))) == 0:
							defs.append(efs2)
							r = self.results[tuple(efs2)]
							source = ''.join(['%s * '%f for f in efs2])[:-3]
							for columnHeader, dictKey in columnHeaderDictKey.items():
					
								for type, suffix in calculationTypes.items():	
									if dictKey == '':
										collectionDict[columnHeader].append(source)
									elif dictKey == 'type':
										collectionDict[columnHeader].append(type)		
									elif dictKey == 'eta':
										collectionDict[columnHeader].append(r[dictKey])																		
									elif dictKey == 'ss':
										collectionDict[columnHeader].append(r[dictKey])
									else:
										valueKey = dictKey+suffix
										if valueKey in r:
											collectionDict[columnHeader].append(r[valueKey])
										else:
											collectionDict[columnHeader].append('-')
							
		
				errorSource ='Error(%s)'%' *'.join([f for f in efs if
                                             f not in bFactors])
				
				
				for columnHeader, dictKey in columnHeaderDictKey.items():
					for type, suffix in calculationTypes.items():
						if dictKey == '':
							collectionDict[columnHeader].append(errorSource)
						elif dictKey == 'type':
							collectionDict[columnHeader].append(type)
						elif dictKey in ['df','ss']:
							if dictKey == 'df':
								collectionDict[columnHeader].append(r['dfe{}'.format(suffix)])
							elif dictKey == 'ss':
								collectionDict[columnHeader].append(r['sse'])

						elif dictKey == 'eps':	
							corKey = '{}{}'.format(dictKey,suffix)
							if corKey != 'eps':
								collectionDict[columnHeader].append(r[corKey])
							elif corKey != 'eps':
								collectionDict[columnHeader].append(r[corKey])
							else:
								collectionDict[columnHeader].append('-')
						elif dictKey == 'mss':
							msKey = 'mse{}'.format(suffix)	
							collectionDict[columnHeader].append(r[msKey])
						else:
							collectionDict[columnHeader].append('')
							
				
		
		withinDataFrame = pd.DataFrame.from_dict(collectionDict)
		return (betweenDataFrame, withinDataFrame), (titleBetween, titleWithin)
		
		
										  
										  
	def get_unique_categorical_values(self, dataframe, factors):
		'''
		'''
		uniqueLevelsDict = dict() 
		numbUniqueLevels = []
		for factor in factors:
			uniqueValues = dataframe[factor].unique()
			uniqueLevelsDict[factor] = uniqueValues
			numbUniqueLevels.append(uniqueValues.size)
		return uniqueLevelsDict, numbUniqueLevels
		
	def num2binvec(self,d,p=0):
		'''
		'''
		d,p=float(d),float(p)
		d=abs(round(d))
		if d==0.:
			b=0.
		else:
			b=[]
			while d>0.:
				 b.insert(0,float(np.remainder(d,2.)))
				 d=np.floor(d/2.)
		return list(np.array(list(np.zeros(int((p-len(b)))))+b)+1)

	
	def matrix_rank(self,arr,tol=1e-8):
		'''
		'''
		arr = np.asarray(arr)
		if len(arr.shape) != 2:
			raise ValueError('Input must be a 2-d array or Matrix object')
		svdvals = scipy.linalg.svdvals(arr)
		return np.sum(np.where(svdvals>tol,1,0))    
	
	def unique_combinations(self,items,n):
		'''
		'''
		if n==0: yield []
		else:
			for i in range(len(items)):
				for cc in self.unique_combinations(items[i+1:],n-1):
					yield [items[i]]+cc   
	
	def observed_power(self,df,dfe,nc,alpha=0.05,eps=1.0):
		'''
		http://zoe.bme.gatech.edu/~bv20/public/samplesize.pdf
       
    	observed_power(3,30,16) should yield 0.916

    	Power estimates of when sphericity is violated require
    	specifying an epsilon value.

    	See Muller and Barton (1989).
    	http://www.jstor.org/stable/2289941?seq=1
		'''
		crit_f = scipy.stats.f(df*eps, dfe*eps).ppf(1.-alpha)
		return scipy.stats.ncf(df*eps, dfe*eps, nc*eps).sf(crit_f)		
		 

   
 ## testing  
#dataFrame = pd.read_csv('testData_anova_within.csv',sep=';') 
#aov = Anova(dataFrame,'ERROR',wFactors=['TIMEOFDAY','COURSE','MODEL'])        
#print(aov.finalResult)        
        		
#Anova(dataFrame, 'WORDS', bFactors=['AGE','CONDITION'])	
	
#Anova(dataFrame, 'SUPPRESSION', wFactors=['CYCLE','PHASE','AGE'],bFactors=['GROUP'])	
		