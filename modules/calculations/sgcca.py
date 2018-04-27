"""
	""SGCCA""
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

"""
Resource - 
Naming of variables etc is based on the author's R package RGCCA (cran).
https://www.ncbi.nlm.nih.gov/pubmed/24550197
"""


import numpy as np 
import pandas as pd
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score



class SGCCA(object):
    
    def __init__(self, X, Y, C = None, n_components = None,
                 c1 = None, scheme = 'horst', scaleData = False,
                 featureNames = None):

        self.params = {}
        
        self.params['scheme'] = scheme
        self.params['scaleData'] = scaleData
        self.params['classes'] = Y
        self.params['n_components'] = n_components
        self.params['featureNames'] = featureNames
        
        #a-priori connection matrix
        self.params['C'] = C
        
        #constraints
        self.params['c1'] = c1
        
        validInput, msg = self.check_input(X,Y)
        if validInput:
            self.create_dummy_y(Y)

            if self.params['scaleData']:
                X = self.scale_data(X)

            self.params['X'] = X
            
        else:
            print(msg)
 

    def initiate_fit(self,X, params):
        '''
        '''        
        if params['n_components'].shape[0] != len(X):
            params['n_components'] = np.append(params['n_components'],[params['n_components'].max()])
        
        params['ndef1'] = params['n_components'] - 1 
        params['N'] = params['ndef1'].max()
        params['J'] = len(X)
        params['pjs'] = [x.shape[1] for x in X]
        
        params['AVE_x'] = []
        params['AVE_outer'] = np.empty(max(params['n_components']))


        if params['C']  is None:
            #if no a priori correlation is given
            params['C']  = np.ones((len(X),len(X)),int)           
            np.fill_diagonal(params['C'],0)

        if params['C'].shape[0] != len(X) and params['C'].shape[1] != len(X):

            if params['C'].shape[0] == len(X)-1:
                
                add = np.asarray([1]*(len(X)-1))
                params['C'] = np.append(params['C'],add.reshape(2,1),axis=1)
                add = np.asarray([1]*len(X)).reshape(1,len(X))
                params['C'] = np.append(params['C'],add,axis=0)
                np.fill_diagonal(params['C'],0)
            else:
                print('Could not interpret C input array.')
                return


        ## check l1 constraints (c1) - cannot be > 1 and < 1/pjs (e.g. number of features)
            
        if len(params['c1'].shape) == 1:# and params['c1'].shape[0] != len(X):

            if params['c1'].size != params['J']:
                params['c1'] = np.append(params['c1'],[1])

            if any(c1 > 1 or c1 < 1/np.sqrt(pjs) for c1,pjs in zip(params['c1'],params['pjs'])):
                print('L1 constraints (c1) must be between 1 and 1/ #features')
                return

        else:
            if params['c1'].shape[1] == params['n_components'].shape[0]:
                pass
            else:
                add = np.ones((params['c1'].shape[1],1))
                params['c1'] = np.append(params['c1'], add,axis=1)

            for n in range(params['c1'].shape[0]):
                if any(c1 > 1 or c1 < 1/np.sqrt(pjs) for c1,pjs in zip(params['c1'][n,:],params['pjs'])):
                    print('L1 constraints (c1) must be between 1 and 1/ #features')
                    return
                else:
                    continue
        ## check number of components - must be smaller than the number of features.                
        if np.where(params['n_components'] - params['pjs'] > 0)[0].size > 0:
            print('Number of components has to be smaller than the number of features.')
            return
        
        return params
    
    def check_input(self,X,Y):
        '''
        '''
        
        if isinstance(X,list) == False:
            msg = "X must be a list of data sets (blocks)"
            return False, msg

        if any(X[0].shape[0] != x.shape[0] for x in X):
            msg = 'Blocks must have the same number of rows.'
            return False, msg
        
        if self.params['n_components'] is None:
            self.params['n_components'] = np.asarray([1]*len(X))
        else:
            self.params['n_components'] = np.asarray(self.params['n_components'])
                                                     
        if any(comp < 1 for comp in self.params['n_components']):
            return False, 'One must compute at least one component per block.'

        if self.params['scheme'] not in ['horst','factorial','centroid']:
                return False, 'Scheme must be horst, factorial or centroid'           


        if isinstance(Y,list):

            if len(Y) != X[0].shape[0]:
                return False, 'Y list shape does not match number of rows in X'

        elif isinstance(Y,np.ndarray):
                                                     
            if Y.shape[0] != X[0].shape[0]:
                return False, 'Y numpy array shape (1st dim) does not match number of rows in X'
        
        return True, ''
                                                     
        
    def fit(self, params = None, save = True):
        '''
        '''
                    
        if params is None:
            params = self.params.copy()
        else:
            if 'X' not in params:
                print('X  not found in params')
                return

        results, params = self.calculate(params)
        if results is None:
            print('Fit failed!')
            return 
        if save:
            ## function get_results needs this object
            self.finalResult = results
            self.params = params
            
        return results 
                

    def calculate(self,params):

            ## add dummy y matrix - going for supervised approach
            
            X = params['X'] + [params['Y']]
            # add parameters to

            params = self.initiate_fit(X,params)
            if params is None:
                return None, None

            if params['N'] == 0:
                
                sgcca = _sgccak(X,params['C'], params['c1'],params['J'],
                                params['pjs'], params['scheme'])
                
                result = sgcca.output
                self.Y = []

                for b in range(params['J']):
                    self.Y.append(result['Y'][:,b])
                ## Avereage Variance Explained (AVE)
                corrColl = []
                for j in range(params['J']):
                    corr = np.apply_along_axis(np.corrcoef,0,X[j],
                                               self.Y[j])[0][-1]
                    corrColl.append(Corr)
                    params['AVE_x'].append( np.mean(corr**2))

                AVE_outer = np.sum(params['pjs'] * np.array(self.AVE_x)) / np.sum(params['pjs'])

                AVE = {'AVE_outer':AVE_outer,
                          'AVE_inner':result['AVE_inner'],
                          'AVE_X':params['AVE_x']}
                
                finalResult = {'Y':[x.reshape(x.size,1) for x in self.Y],
                               'a':[x.reshape(x.size,1) for x in  result['a']],
                               'n_components':self.n_components,
                               'scheme':self.scheme,'cl' : self.cl,'AVE':AVE,
                               'X':self.X,'corr':corrColl}
                
                self.finalResult = finalResult
                
                return finalResult

            else:
                
                R = X
                ## initiate 
                Y,P, a, astar, crit = [], [], [], [], []
                nb_ind = X[0].shape[0]
                AVE_inner = np.empty(max(params['n_components']))
                P = [np.empty((params['pjs'][J],params['N']+1)) for J in range(params['J'])]
                a  =[np.empty((params['pjs'][J],params['N']+1)) for J in range(params['J'])]
                astar = [np.empty((params['pjs'][J],params['N']+1)) for J in range(params['J'])]

                Y = [np.empty((nb_ind ,params['N']+1)) for J in range(params['J'])]

                
                for n in range(1,params['N']+1):                    
                    if len(params['c1']) == params['J']:

                        c1 = params['c1']
                    else:
                        c1 = params['c1'][n-1,:]
                        
                        
                    #perform sgcca 
                    result = _sgccak(R,params['C'],c1,params['J'],
                                     params['pjs'], params['scheme']).output
                    
                    AVE_inner[n-1] = result['AVE_inner']
                    crit.append(result['crit'])

                    for b in range(params['J']):
                        Y[b][:,n-1] = result['Y'][:,b]
                        
                    #deflatation 
                    deflaResult = self.defl_select(result['Y'],R,
                                                   params['ndef1'],n,
                                                   nbloc=params['J'])

                    R = deflaResult['resdef1']

                    for b in range(params['J']):
                        P[b][:,n-1] = deflaResult['pdef1'][b]
                        a[b][:,n-1] = result['a'][b]
                    if n == 1:
                        for b in range(params['J']):
                            astar[b][:,n-1] = result['a'][b]
                    else:
                        for b in range(params['J']):
                            m1 = np.matmul(a[b][:,n-1],P[b][:,0:n-1])
                            m2 = np.matmul(astar[b][:,0:n-1],m1)
                            astar[b][:,n-1] =  result['a'][b] - m2

                print('computing of the SGCCA block components')

                if len(params['c1']) == params['J']:
                    cl = params['c1']
                else:
                    cl = params['c1'][params['N']]


                sgcca = _sgccak(R,params['C'], cl,
                                params['J'], params['pjs'],
                                params['scheme'])
                result = sgcca.output

                
                crit.append(result['crit'])
                AVE_inner[-1] = result['AVE_inner']
                
                for b in range(params['J']):
                    
                    Y[b][:,params['N']] = result['Y'][:,b]
                    a[b][:,params['N']] = result['a'][b]
                    m1 = np.matmul(a[b][:,params['N']], P[b][:,0:params['N']])
                    m2 = np.matmul(astar[b][:,0:params['N']],m1)
                    astar[b][:,params['N']] =  result['a'][b] - m2


                coeffs = []
               
                for j in range(params['J']):
                    saveCorr = np.empty((X[j].shape[1],Y[j].shape[1]))
                    for m in range(Y[j].shape[1]):
                        
                        saveCorr[:,m] = np.apply_along_axis(np.corrcoef,0,X[j],Y[j][:,m])[0][-1]
                        
                    coeffs.append(saveCorr)                                         
                    mean_ = np.mean(saveCorr**2,axis=0)
                    params['AVE_x'].append(mean_.reshape(mean_.size,1))

                outer = np.concatenate(params['AVE_x'],axis=1)#.reshape(max(self.n_components),self.J)
                
                AVE_outer = []
                for j in range(0,max(params['n_components'])):
                               AVE_outer.append(np.sum(params['pjs'] * outer[j,:])/np.sum(params['pjs']))
                AVE = {'AVE_X' : params['AVE_x'],
                    'AVE_outer' : AVE_outer,
                    'AVE_inner' : AVE_inner}
            
 
                finalResult = {'Y': Y,'C':params['C'],'astar':astar,'a':a,
                        'scheme':params['scheme'],'crit':crit,'AVE':AVE,
                        'n_components':params['n_components'],'X':X,'corr':coeffs}                   
                                
                return finalResult, params


            

    def tune(self, tuneParams, nSplits = 2, repeat = 2, test_size = 0.2):

        if hasattr(self,'params') == False:
            print('Perform fit() first')
            return
        
        if test_size > 0.9 or test_size < 0.1:
            print('test_size must be between 0.1 and 0.9')
            return
        if isinstance(repeat,int) == False:
            print('Repeat must be an int. ')
            return
        if isinstance(nSplits,int) == False:
            print('nSplits must be an int')
            retur
        if isinstance(tuneParams,dict) == False:
            print('tuneParams must be provided as a dict.')
            return

        if nSplits > np.unique(self.params['classes']).size:
            print('Number of Splits cannot be greater than the number of classes.')
            return

        # original data
        X = self.params['X'] + [self.params['Y']]


        if any(key not in self.params for key in tuneParams.keys()):
            
            print('Unknown parameter(s) given in tuneDict. Will be ignored')

        tuneValues = []
        tuneParam = []
        
        for key,values in tuneParams.items():
            ## check if given tune dict can be interpreted
            if isinstance(values,list) == False or key not in self.params:
                print('Ignoring .. {}. Not found as a parameter ..'.format(key))
                continue
            else:

                tuneParam.append(key)
                tuneValues.append([np.asarray(value) for value in values])
                
        tuneGrid = itertools.product(*tuneValues)

        # result collection
        resultColl = {'id':[],'repeat':[],'nSplit':[],
                      'error':[]}
        sgccaColl = {}
        
        id_ = 0
        for search in tuneGrid:

            params = self.params.copy()
            ##update parameter for search
            for n,key in enumerate(tuneParam):
                params[key] = search[n]
            ## start loopf over repeats
            for rep in range(repeat):
                # initiate cross validation                   
                rskf = StratifiedShuffleSplit(n_splits=nSplits, test_size = test_size)
                nSplit = 0
                
                for train_idx, test_idx in rskf.split(X[0],params['classes']):

                    Xtrain = [x[train_idx,:] for x in self.params['X']]
                    Xtest = [x[test_idx,:] for x in self.params['X']]

                    params['X'] = Xtrain
                    params['Y'] = self.params['Y'][train_idx,]
                    ## calculate sggca
                    sggcaResults,_ = self.calculate(params)
                
                    ## project test data
                    XtestProj = self.project_test_data(Xtest,sggcaResults)
                    
                    # -1 excludes the outcome block
                    XtrainProjConc = np.concatenate(sggcaResults['Y'][:-1], axis=1)
                    XtestProjConc = np.concatenate(XtestProj,axis = 1)

                    err = self.discriminant_analysis(XtrainProjConc,
                                                     XtestProjConc,
                                                     params['classes'],
                                                     train_idx, test_idx)

                    resultColl['error'].append(err)
                    resultColl['id'].append(id_)
                    resultColl['nSplit'].append(nSplit)
                    resultColl['repeat'].append(rep)

                    sgccaColl[id_] = [params.copy(),sggcaResults]
                    print(params['c1'])
                    self.get_non_zero_features(sggcaResults, index = train_idx) 
                    
                    id_ += 1
                    nSplit += 1
                   
        cvResult = pd.DataFrame.from_dict(resultColl)
        # reset cl
        print(cvResult)
        return cvResult

    def get_non_zero_features(self, sggcaResult = None,
                              getNames = True, index = None,
                              blockNames = None):

        

        if sggcaResult is None:
            sggcaResult = self.finalResult
        if getNames:
            sggcaResult = self.get_result(sggcaResult, index = index)
        A = sggcaResult['a']
        C = sggcaResult['corr']
        C = [pd.DataFrame(c, columns = ['comp_{}'.format(n+1) for n in range(c.shape[1])]) for c in C]
       # nonZeroA = []

        for n,a in enumerate(A):
            idx_ = np.where(np.sum(np.abs(a),axis=1)>0)[0]
            nonZeroFeatures = a.iloc[idx_,:]
            nonZerC = C[n].iloc[idx_,:]
            if n == 0:
            	nonZeroFeatures.loc[:,'Feature'] = nonZeroFeatures.index
            	nonZerC.loc[:,'Feature'] = nonZeroFeatures.index
            	if blockNames is not None:
            		nonZeroFeatures.loc[:,'Block'] = [blockNames[n]] * len(nonZeroFeatures.index)
            		nonZerC.loc[:,'Block'] = [blockNames[n]] * len(nonZeroFeatures.index)
            	outA = nonZeroFeatures
            	outC = nonZerC
            	
            else:
            	nonZeroFeatures.loc[:,'Feature'] = nonZeroFeatures.index
            	nonZeroFeatures.loc[:,'Block'] = [blockNames[n]] * len(nonZeroFeatures.index)
            	nonZerC.loc[:,'Feature'] = nonZeroFeatures.index
            	nonZerC.loc[:,'Block'] = [blockNames[n]] * len(nonZeroFeatures.index)
            	outA = outA.append(nonZeroFeatures, ignore_index = True)
            	outC = outC.append(nonZerC, ignore_index = True)
			         
            #nonZeroA.append(nonZeroFeatures)
			
            n = len(nonZeroFeatures.index)

        return outA, outC
            
                   
    def eval_comps(self,splits = 2,repeatCV = 2, ncomp = 2, ignoreC1 = True):
        '''    
        '''
 #       self.initiate_fit(X)
        if 'X' not in self.params:
            print('X not found. Entered valid input in init.?')
            return
        params  = self.params.copy()
        
        # get original X and Y to subset splits in CV
        X = params['X'] + [params['Y']]

        if params['n_components'].max() <= ncomp and ignoreC1 == False:
            ignoreC1 = True
            print("Entered constraints (c1) are ignored and set to 1.")
            
 
        params['n_components'] = np.asarray([ncomp] * params['J'])
       
        rskf = RepeatedStratifiedKFold(splits,repeatCV)
        n = 0
        
        nSplit = 0
        repeat = 1
        
        # result collection
        resultColl = {'id':[],'repeat':[],'nSplit':[],
                      'comps':[],'error':[]}
        
        # cross validation 
        for train_idx,test_idx in rskf.split(params['X'][0],params['classes']):
            
            if ignoreC1:
                #assume 1 - no sparsity
                params['c1'] = np.asarray([1]*len(params['X']))

            # keep track on splits/repeats 
            if n % splits == 0 and nSplit != 0:
                repeat += 1
                nSplit = 0
                
            ## import, taking X from initialization. 
            Xtrain = [x[train_idx,:] for x in self.params['X']]
            Xtest = [x[test_idx,:] for x in self.params['X']]

            params['X'] = Xtrain
            params['Y'] = self.params['Y'][train_idx,]
            
            sggcaResults,_ = self.calculate(params)
            
            ## project test data
            XtestProj = self.project_test_data(Xtest,sggcaResults)
            
            # concatenate block values
            # -1 excludes the outcome block
            XtrainProjConc = np.concatenate(sggcaResults['Y'][:-1], axis=1)
            XtestProjConc = np.concatenate(XtestProj,axis = 1)

            # calculate error by adding components step-wise
            for compRange in range(1,ncomp+1):
                colRange = range(0,compRange)
                finalIdx = []
                
                # generate index lists to subset concatenated
                # transformed X data.
                for j in range(params['J']-1):
                    finalIdx.extend([colIdx + (j*ncomp) for colIdx in colRange])
                    
                # predcit classs labels
                err = self.discriminant_analysis(XtrainProjConc[:,finalIdx],
                                                 XtestProjConc[:,finalIdx],
                                                 params['classes'],
                                                 train_idx, test_idx)
                
                # append error and other data to lists in dict
                resultColl['repeat'].append(repeat)
                resultColl['nSplit'].append(nSplit)
                resultColl['id'].append(n)
                resultColl['comps'].append(compRange)
                resultColl['error'].append(err)
                
                n += 1
            
            nSplit += 1
            
        cvResult = pd.DataFrame.from_dict(resultColl)
        # reset cl
        return cvResult
        

    def discriminant_analysis(self,XtrainProj,XtestProj,
                              Y,train_idx, test_idx):

            
        LDA = LinearDiscriminantAnalysis()
        LDA.fit(XtrainProj,Y[train_idx])
        
        predictedClasses = LDA.predict(XtestProj)
        trueClasses = Y[test_idx].tolist()
        falsePredictions = [x for x in predictedClasses if \
                            trueClasses[predictedClasses.tolist().index(x)] != x]
        prec = precision_score(trueClasses,predictedClasses,
                               average = 'weighted')
        
        return len(falsePredictions)/len(trueClasses)

    def project_test_data(self,Xtest,sggcaResults):
                          
            XtestTransformed = []
            for q in range(len(X)):
                yy = np.empty((Xtest[q].shape[0],
                               sggcaResults['a'][q].shape[1]))
                
                for m in range(sggcaResults['a'][q].shape[1]):
                    
                    yy[:,m] = np.apply_along_axis(self.cross,1,\
                                         Xtest[q],sggcaResults['a'][q][:,m])
                    
                XtestTransformed.append(yy) 

                

            return XtestTransformed

                          
    def scale_data(self,X):
        '''
        Scales by removing mean and scaling data to unit variance
        '''
        scaledX = []
        for x in X:
            x = StandardScaler().fit_transform(x)            
            scaledX.append(x)
        return scaledX
  
        
    def defl_select(self,yy,rr,nncomp,n,nbloc):
        resdef1 = []
        pdef = []
        for q in range(nbloc):
            if n <= nncomp[q]:
                defltmp = self.deflation(rr[q],yy[:,q])
                resdef1.append(defltmp['R'])
                pdef.append(defltmp['p'])
            else:
                resdef1.append(rr[q])
                pdef.append(np.zeros(rr[q].shape[1]))
        return {'resdef1':resdef1,'pdef1':pdef}

    def get_result(self, sggcaResult = None, index = None):

        if sggcaResult is None:
            if hasattr(self,'finalResult'):
                sggcaResult = self.finalResult
            else:
                print('Specify sggcaResult object..')
                return 

        
        # modify results
        yMod = []
        for p,Y in enumerate(sggcaResult['Y']):
                columnNames = ['comp_{}'.format(n+1) \
                               for n in range(self.params['n_components'][p])]
                if index is None:
                    idx_ = self.params['classes']
                else:
                    idx_ = self.params['classes'][index]
                    
                yMod.append(pd.DataFrame(Y,columns=columnNames,
                                         index = idx_))

                    
        sggcaResult['Y'] = yMod
        

        if self.params['featureNames'] is not None:
            aMod = []
            astarMod = []
            
 			
            for p, a in enumerate(sggcaResult['a']):
            	
                if len(self.params['featureNames']) == len(sggcaResult['a'])\
                or p != len(sggcaResult['a'])-1:
                    
                    rowNames = self.params['featureNames'][p]
                    columnNames = ['comp_{}'.format(n+1) for n in range(a.shape[1])]
                else:
                    break
                
                aMod.append(pd.DataFrame(a,index=rowNames,columns=columnNames))
                astarMod.append(pd.DataFrame(sggcaResult['astar'][p],
                                index = rowNames,columns=columnNames))
			
			
			
            sggcaResult['a'] = aMod
            sggcaResult['astar'] = astarMod                   

        return sggcaResult
        
                               
    def deflation(self,X,y):

        p = np.apply_along_axis(self.cross,0,
                                X,y) / self.cross(y,y)
        R = X - np.outer(y,p)
        return {'R':R,'p':p}
         
    def cross(self,x,y):

        return np.sum(x*y)
        
    def create_dummy_y(self,Y):
        uniqueVals  = np.unique(Y)
        nClasses = uniqueVals.size
        Ydummy = np.zeros((Y.shape[0],nClasses))
        orderVals = []
        for n,target in enumerate(Y):
            col = np.where(uniqueVals==target)
            Ydummy[n,col] = 1
            orderVals.append(target)
            
        self.params['Y'] = Ydummy
        self.params['classOrder'] = orderVals		
	


class _sgccak(object):
        
    
    def __init__(self,X,C,cl,J,pjs,scheme = 'centroid', init = 'svd',
                 bias=True, tol = 1e-10):
        
        self.J = J
        self.pjs = pjs
        self.scheme = scheme
        self.C = C
        self.cl = cl 
        self.A = X ## this naming was used in the sgccak
        self.tol = tol
        self.AVE_x = [0] * self.J

        output = {}

        # initiate a 
        if init == 'random':
            self.a  = [np.random.randn(n) for n in self.pjs]
        elif init == 'svd':
            
            try:
                self.a = [np.linalg.svd(x)[-1].T[:,0] for x in self.A]
            except:
                self.a  = [np.random.randn(n) for n in self.pjs]

        # empty vectors for Y and Z 
        self.Y = np.zeros((X[0].shape[0],self.J))
        self.Z = np.zeros((X[0].shape[0],self.J))

        self.fit(bias) 



    def fit(self,bias):

        # Apply constraints of the general optimization problem
        # compute the outer components
        
        # l1 constraint

        self.const = self.cl*np.sqrt(self.pjs)
                        
        iter_ = 0
        self.crit = []

        # initiliaze
        for q in range(self.J):
                data = self.A[q]
                self.Y[:,q] = np.apply_along_axis(self.cross,1,
                                          data,self.a[q])
                
                self.a[q] = self.soft_threshold(self.a[q], self.const[q])
                self.a[q] = self.a[q]/self.norm2(self.a[q])
        self.old_a = self.a.copy()

        
        self.crit_old = np.sum(self.C * (np.diagonal(self.cov2(self.Y,bias=bias))))
        while iter_ < 1000:
            break_ = self.calculation(True,iter_)
            iter_ += 1
            if break_:
                break
        if iter_ > 1000:
            print("SGCCA algorithm did not converge after 1000 iterations.")
        corY = pd.DataFrame(self.Y).corr().values
        AVE_inner = np.sum(self.C * corY **2 / 2) / (np.sum(self.C)/2)

        self.output = {'AVE_inner': AVE_inner,
                       'Y': self.Y,
                       'C': self.C,
                       'a':self.a,
                       'scheme':self.scheme,
                       'crit': [crit for crit in self.crit if crit != 0],
                       'cl':self.cl,
                       'const': self.const}
        
            
    def calculation(self, bias, iter_):
        '''
        Performs calculation
        '''
        for q in range(self.J):
            if self.scheme == 'horst':
                CbyCovq = self.C[q,] 
            elif self.scheme == 'factorial':
                CbyCovq = self.C[q,]*2*self.cov2(self.Y,self.Y[:,q],bias = bias)
            elif self.scheme == 'centroid':
                CbyCovq = self.C[q,]*np.sign(self.cov2(self.Y,self.Y[:,q],bias = bias))

            # re calculate Z, a, and Y 
            mult = np.multiply(CbyCovq,self.Y)
            self.Z[:,q] = np.sum(mult,axis=1)

            self.a[q] = np.apply_along_axis(self.cross,0,self.A[q],self.Z[:,q])
            self.a[q] = self.soft_threshold(self.a[q],self.const[q])
            self.a[q] = self.a[q]/self.norm2(self.a[q])
            self.Y[:,q] = np.apply_along_axis(self.cross,0,self.A[q].T,self.a[q])

        # save crit
        self.crit.append(np.sum(self.C * (np.diagonal(self.cov2(self.Y,bias=bias)))))

       # print(self.Y)
        
        if np.abs(self.crit[iter_]-self.crit_old) < self.tol:
            return True
        self.crit_old = self.crit[iter_]
        self.old_a = self.a
        return False 
        
    def cov2(self,x,y=None,bias=False,rowvar=False):
        
        cov = np.cov(x,y,rowvar,bias)
        return cov
    
    def cross(self,x,y):

        return np.sum(x*y)

        
    
    def norm2(self,arr):
        
        a = np.sqrt(np.sum(arr**2))
        
        if a== 0: 
            a = .05
        return a        

    def soft_threshold(self,x,sumabs=1):
        thresh = self.soft(x,self.binary_search(x,sumabs))
        return thresh
        

    def soft(self,x,d):
        
        return np.sign(x)*np.maximum(0,np.abs(x)-d)
    

        
    def binary_search(self,argu,sumabs):
        if self.norm2(argu) == 0 or np.sum(np.abs(argu/self.norm2(argu))) <= sumabs:
            return 0
        lam_max = np.max(abs(argu))
        lam1 = 0
        lam2 = lam_max
        iter_ = 0
        
        while iter_ < 500:
            su = self.soft(argu, (lam1+lam2)/2)
            if np.sum(np.abs(su/self.norm2(su))) < sumabs:
                lam2 = (lam1+lam2)/2
            else:
                lam1 = (lam1+lam2)/2

            if (lam2+lam1)/lam1 < 1e-10:

                if lam2 != lam_max:
                    return lam2
                else:
                    return lam1
                
            iter_ += 1
        return (lam1+lam2)/2
             
       
# 
# if __name__ == "__main__":
# 
#         
# 	c = np.random.rand(50, 70)
# 	d = np.random.rand(50, 80)
# 
# 	c = pd.read_table("~/Downloads/protTom.txt")
# 	d = pd.read_table("~/Downloads/metabol_Tom_trans.txt")
# 
# 
# 
# 	X = [d[[col for col in d.columns if col != 'Index']].values,
#      c[[col for col in c.columns if col != 'Index']].values]        
# 
# def corr2(df1, df2):
#     n = len(df1)
#     v1, v2 = df1.values, df2.values
#     sums = np.multiply.outer(v2.sum(0), v1.sum(0))
#     stds = np.multiply.outer(v2.std(0), v1.std(0))
#     return pd.DataFrame((v2.T.dot(v1) - sums / n) / stds / n,
#                         df2.columns, df1.columns)
# 
# #corr(df1, df2)
# 
# x1 = pd.DataFrame(X[0])                
# x2 = pd.DataFrame(X[1])
# corr = corr2(x1,x2)
# #print(corr)
# 
# 
# 
# 
# C = np.array([[0,0.1],[0.1,0]])
# 
# C1 = np.array([[0,0.2],[0.2,0]])
# print(C)
# 
# c1 = np.array([0.1,0.1])
# print(c1)
# names = [[col for col in d.columns if col != 'Index'],
#          [col for col in c.columns if col != 'Index']
#          ]
# model = SGCCA(X = X, Y = d['Index'], C = C, 
#               n_components = [2,2],c1=c1,
#               featureNames = names)
# 
# 
# 
# model.fit()
# model.get_non_zero_features()
# #tune(self, tuneDict, test_size = 0.2):
# 
# res = model.tune({'c1':[[0.1,0.1],[0.2,0.1],[0.3,0.2]]})
# 
# 
# print(res)
# print(Hello)
# #print(model.eval_comps(splits = 2,repeatCV = 5,
# #                       ncomp = 4, ignoreC1 = True))
# fit = model.get_result()
# 
# print(fit)
# 
# 
# sum_ = np.sum(fit['a'][1],axis=1)
# cols = c.columns[np.where(sum_!=0)[0]]
# 
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# f1=plt.figure()
# 
# 
# fit['a'][0].to_csv("tomMet.txt")
# 
# fit['a'][1].to_csv("tomProt.txt")
# 
# print(fit['Y'][1])
# 
# for n in range(2):
#     sum_ = np.sum(fit['a'][n],axis=1)
#     idx = np.where(sum_!=0)[0]
#     ax = f1.add_subplot(2,2,n+1)
#     ax.scatter(fit['Y'][n]['comp_1'],
#                fit['Y'][n]['comp_2'],c = sns.color_palette('Blues',20))
# plt.show()
# 


        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
   
        
        
        
        
        
        
