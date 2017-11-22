# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:31:24 2017

@author: christopherhedenberg
"""

class data_cleaner():
    def __init__(self,x=None,y=None,resp=None,cv=None,split=False,save_data=False,data_path=None,split_by=None,split_parent = None,nbins=5,seed = 123456789):
        self.split=split
        self.resp=resp
        self.x=x
        self.split_by=split_by
        self.split_parent=split_parent
        self.nbins=nbins
        self.seed=seed
        self.y=y
        self.kfold=None
        self.folds=None
        self.save_data=save_data
        self.data_path=data_path
        if self.split==True:
            self.cv="TestInd"
        else:
            self.cv=cv
        
    
    def split_data(self):
        try:
            from sklearn import model_selection
            if self.split_by==None and self.split_parent==None:
                self.kfold=model_selection.KFold(n_splits=self.nbins,random_state=self.seed)
                self.folds=self.kfold.split(self.x)
                self.folds = self.kfold.split(self.x)
            elif self.split_by == None and self.split_parent!=None:
                self.kfold=model_selection.GroupKFold(n_splits=self.nbins,random_state=self.seed)
                self.folds=self.kfold.split(self.x,groups=self.split_parent)
            elif self.split.by !=None and self.split_parent==None:
                self.kfold=model_selection.StratifiedKFold(n_splits=self.nbins,random_state=self.seed)
                self.folds=self.kfold.split(self.x,self.split_by)
            i=0
            for train,test in self.folds:
                self.x.loc[test,"TestInd"]=i
                i+=1
        except:
            ValueError("X data is missing for splitting")
        

    
    def clean_data(self):
        if self.save_data==True:
            outfile=raw_input("What would you like to name the data")
        if self.split == True:
            self.split_data()
        #self.x = self.x.impute(split_by="TestInd")
        #self.x = self.x.impute(split_by=self.cv,num_mthd='med',cat_mthd='top')
        #self.x = self.x.impute(split_by=self.cv,cat=False,num_mthd='mode')
        #self.x = self.x.weighted_bucket(nbin=3)
        #self.x = self.x.weighted_bucket(nbin=5)
        #self.x = self.x.weighted_bucket(nbin=10)
        #self.x = self.x.weighted_bucket(nbin=20)    
        if self.save_data==True:
            if self.data_path==None:
                self.x.to_pickle("%s.p"%outfile)
            else:
                self.x.to_pickle("%s.p"%(self.data_path+"/"+outfile))


            