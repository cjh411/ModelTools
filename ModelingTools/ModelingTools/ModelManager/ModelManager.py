# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 08:53:34 2017

@author: christopherhedenberg
"""

#from Utilities.Utilities import *
import numpy as np


class Model_Manager():
    def __init__(self,models=['glm','enet'],family="tweedie",loss=tweedie_deviance,x=None,y=None):
        self.loss=loss
        self.models=models
        self.x=x
        self.y=y
        self.family=family
        self.model=None
        self.loss_dict={"gaussian":None,"poisson":poisson_deviance,"gamma":gamma_deviance,"tweedie":tweedie_deviance}
        
        
class Model():
    def __init__(self,family,x,y,offset=None,exposure=None,weight=None,mdl_by=None):
        self.family=family
        self.x=pd.DataFrame(x)
        self.y=pd.DataFrame(y)
        self.offset=offset
        self.exposure=exposure
        self.weights=weight
        self.mdl_by=mdl_by
        self.numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        if self.mdl_by==None:
            self.mdl_by="All"
            self.mdl_splits=["All"]
            self.x["All"]="All"
        else:
            try:
                self.mdl_splits=list(set(self.x[mdl_by].tolist()))
            except:
                ValueError("split_by variable not found in x dataset")
        self.predictors=self.get_predictors()
                
    def get_dummies(self,mdl_vars=None,include_split=None,exclude_split=None):
        if type(mdl_vars) == type(None):
            mdl_vars=self.predictors
        if type(include_split)!=type(None) and type(exclude_split)!=type(None):
            raise TypeError("Only include or exclude split may be used")
        if type(include_split)!=type(None):
            allcols = pd.DataFrame(self.x[mdl_vars])
            allcols = pd.DataFrame(allcols.loc[self.x[self.mdl_by]==include_split,mdl_vars])
        elif type(exclude_split)!=type(None):
            allcols = pd.DataFrame(self.x[mdl_vars])
            allcols = pd.DataFrame(allcols.loc[self.x[self.mdl_by]!=exclude_split,mdl_vars])            
        else:
            allcols = pd.DataFrame(self.x[mdl_vars])
        numcol=allcols.select_dtypes(include=[np.number])
        ccol=len(allcols.select_dtypes(exclude=[np.number]).columns.values)
        if ccol>0:
            chardum=pd.get_dummies(allcols.select_dtypes(exclude=[np.number]))
        if len(numcol.columns.values)==0:
            return chardum
        elif ccol==0:
            return numcol
        else:
            return pd.concat([numcol,chardum],axis=1)
        
    def get_predictors(self):
        if type(self.x)==type(pd.Series()):
            allvar=[self.x.name]
            if len(allvar)==0 or self.x.isnull().any()==True:
                raise ValueError("X is either missing columsn or has NaN in single column")
        elif type(self.x)==type(pd.DataFrame()):
            allvar=self.x.columns[~self.x.isnull().any()].tolist()
            if len(allvar)==0:
                raise ValueError("All columns in X have missing values")
        else:
            raise TypeError("Unknown format for x input data. Must be pandas series or data frame")
        print(allvar)
        print(self.mdl_by)
        return [item for item in allvar if item !=self.mdl_by]  
        
        
class glm(Model):
    import statsmodels.api as sm
    def __init__(self,family,x,y,offset=None,exposure=None,weight=None,mdl_by=None):
        Model.__init__(self,family,x=x,y=y,offset=offset,exposure=exposure,weight=weight,mdl_by=mdl_by)
        self.__models__={}
        self.__loss__={}
        self.__gini__={}
        
    def fit_mdl(self,x=None,y=None,p=1.5):
        if type(x)==type(None):
            x=self.get_dummies()
        if type(y) ==type(None):
            y=self.y
        if type(self.x)!=type(None) and type(self.y) != type(None):
            if self.family=="tweedie":
                model=self.sm.GLM(y,x,family=self.sm.families.Tweedie(var_power=p),offset=self.offset,exposure=self.exposure,freq_weights=self.weights).fit()
            elif self.family=="gamma":
                model=self.sm.GLM(y,x,family=self.sm.families.Gamma(),offset=self.offset,exposure=self.exposure,freq_weights=self.weights).fit()
            elif self.family=="poisson":
                model=self.sm.GLM(y,x,family=self.sm.families.Poission(),offset=self.offset,exposure=self.exposure,freq_weights=self.weights).fit()
            elif self.family=="gaussian":
                model=self.sm.GLM(y,x,family=self.sm.families.Gaussian(),offset=self.offset,exposure=self.exposure,freq_weights=self.weights).fit()
            else:
                raise ValueError("Only Distributions supported are tweedie, poisson, gamma and gaussian")
        else:
            raise ValueError("GLM model fitting requires x and y data")
        return model
    
    def fit_list(self,x=None,y=None,varlst=None,p=1.5):
        mdls=[]
        if type(varlst)==type(None):
            varlst = []
        else:
            varlst = list(varlst)
        for var in self.predictors:
            if var not in varlst:
                tmp=[]
                for splitid in self.mdl_splits:
                    print()
                    mdlsvars=varlst+[var]
                    print(mdlsvars)
                    tmp.append((mdlsvars,self.fit_mdl(self.get_dummies(mdlsvars,exclude_split=splitid),self.y[self.x[self.mdl_by]!=splitid],p=p)))
                mdls.append(tmp)
        return mdls
        
    #set this up to return the weights if needed for evaluation    
    def cv_predict_oof(self,dsn,mdl):
        if len(mdl)!=len(self.mdl_splits):
            raise TypeError("Longth of model array is not equal to length of cross validation array")
        preds=[]
        acts=[]
        for ind, splitid in enumerate(self.mdl_splits):
            mdlsvars=mdl[ind][0]
            mdlobj=mdl[ind][1]
            print(mdlsvars)
            #predict method neeeds a way to handle values not seen in training dataset
            preds.append(mdlobj.predict(self.get_dummies(mdlsvars,include_split=splitid)))
            acts.append(self.y[self.x[self.mdl_by]==include_split])
        return acts,preds

#set this up to run many evaluations if needed
        def cv_loss(self,dsn,mdl,loss=gini):
        act,pred=self.cv_predict_oof(dsn,mdl)
        ls = loss()
    
    def forward_select():
        pass

    
            
            
    
        
        
            
            