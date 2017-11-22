# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 18:45:27 2017

@author: christopherhedenberg
"""

import pandas as pd
import wquantiles
import numpy as np



def sort_group(df,sort_by,sort_by_wt=None,bucket_by=None,ascending=True,nbin=10):
    if type(sort_by)!=type('str'):
        raise TypeError("Sortby field must by specified as a string")
    if type(df) != type(pd.DataFrame()) or type(df)!= type(pd.Series()):
        df = pd.DataFrame(df)
    if type(sort_by_wt)!=type(None) and type(sort_by_wt)!=type('str'):
        raise TypeError("sort_by_wt must be a string input")
    elif type(sort_by_wt)==type(None):
        df["Sorter"]=df[sort_by]
    else:
        unique = df[sort_by_wt].unique()
        if 0 in unique or df[sort_by_wt].isnull().any() or np.isinf(df[sort_by_wt]).any():
            raise ValueError("sort_by_wt contains nan, inf or zero")
        else:
            df["Sorter"] = df[sort_by]/df[sort_by_wt]
    if type(bucket_by)==type(None):
        df["Bucketer"] = 1
    else:
        df["Bucketer"]=df[bucket_by]
    df["Group"]=weighted_cuts(df["Sorter"],df["Bucketer"],nbin)
    return df["Group"]

def gini_table(df=None,act_loss=None,pred_loss=None,loss_wt=None,exposure_wt=None,nbin=100):
    if type(df)==type(None) or type(act_loss)==type(None) or type(pred_loss)==type(None):
        raise TypeError("Argument is missing df, act_loss, or pred_loss")
    elif type(act_loss)!=type('str') or type(pred_loss)!=type('str'):
        raise TypeError("act_loss and pred_loss must be string inputs")
        
    if type(loss_wt)==type(None):
        df["Loss_Weight"]=1
    else:
        if type(loss_wt)!=type('str'):
            raise TypeError("loss_wt must be a string")
        df["Loss_Weight"] = df[loss_wt]
    if type(exposure_wt)==type(None):
        df["Exp_Weight"]=1
    else:
        if type(loss_wt)!=type('str'):
            raise TypeError("loss_wt must be a string")
        df["Exp_Weight"] = df[exposure_wt]
    df["Group"]=sort_group(df,pred_loss,"Loss_Weight","Exp_Weight",True,100)
    gini = df[["Group",act_loss]].groupby(["Group"]).sum()
    gini["Sorter"] = gini.index
    gini["Sorter"] = gini["Sorter"].astype('str').str.split(",",expand=True).loc[:,0].str.replace("(","").astype('float')
    gini.sort("Sorter",inplace=True)
    gini = gini.drop(["Sorter"],axis=1)
    gini = gini.cumsum()
    gini["GroupNo"] = 1
    gini["GroupNo"]=gini["GroupNo"].cumsum()
    gini["PctRandom"]=gini["GroupNo"]/nbin
    gini["PctLoss"]=gini[act_loss]/max(gini[act_loss])
    return gini
    
def gini(df=None,act_loss=None,pred_loss=None,loss_wt=None,exposure_wt=None,nbin=100):
    gini = gini_table(df=df,act_loss=act_loss,pred_loss=pred_loss,loss_wt=loss_wt,exposure_wt=exposure_wt,nbin=nbin)
    gini_num = sum(gini["PctRandom"]-gini["PctLoss"])/sum(gini["PctRandom"])
    return gini_num
    
def lift_table(df=None,act_loss=None,pred_loss=None,loss_wt=None,exposure_wt=None,nbin=10):
    if type(df)==type(None) or type(act_loss)==type(None) or type(pred_loss)==type(None):
        raise TypeError("Argument is missing df, act_loss, or pred_loss")
    elif type(act_loss)!=type('str') or type(pred_loss)!=type('str'):
        raise TypeError("act_loss and pred_loss must be string inputs")
        
    if type(loss_wt)==type(None):
        df["Loss_Weight"]=1
    else:
        if type(loss_wt)!=type('str'):
            raise TypeError("loss_wt must be a string")
        df["Loss_Weight"] = df[loss_wt]
    if type(exposure_wt)==type(None):
        df["Exp_Weight"]=1
    else:
        if type(loss_wt)!=type('str'):
            raise TypeError("loss_wt must be a string")
        df["Exp_Weight"] = df[exposure_wt]
    df["Group"]=sort_group(df,pred_loss,"Loss_Weight","Exp_Weight",True,nbin)
    df["%s_a"%act_loss]=df[act_loss]
    df["%s_p"%pred_loss]=df[pred_loss]
    df = df.drop([act_loss,pred_loss],axis=1)
    act_loss = "%s_a"%act_loss
    pred_loss = "%s_p"%pred_loss
    lift = df[["Group",act_loss,pred_loss,"Loss_Weight","Exp_Weight"]].groupby(["Group"]).sum()
    lift["Sorter"] = lift.index
    lift["Sorter"] = lift["Sorter"].astype('str').str.split(",",expand=True).loc[:,0].str.replace("(","").astype('float')
    lift.sort("Sorter",inplace=True)
    lift = lift.drop(["Sorter"],axis=1)
    lift["%s_avg"%act_loss] = lift[act_loss]/lift["Exp_Weight"]
    lift["%s_avg"%pred_loss] = lift[pred_loss]/lift["Exp_Weight"]    
    return lift[["Exp_Weight","Loss_Weight","%s_avg"%pred_loss,"%s_avg"%act_loss]]    
    
def dual_lift_table(df=None,act_loss=None,pred_loss1=None,pred_loss2=None,exposure_wt=None,nbin=10):    
    if type(pred_loss2)!=type('str'):
        raise TypeError("pred_loss2 must be supplied in string format for dual lift")
    dl = lift_table(df=df,act_loss=act_loss,pred_loss=pred_loss1,loss_wt=pred_loss2,exposure_wt=exposure_wt,nbin=nbin)
    dl.columns=["Exposure","Predicted2","Predicted1","Actual"]
    return dl
            
def tweedie_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev  
  
def gamma_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    
    
def poisson_deviance(y,mu,p):
    dev = 2*( ( (y**(2-p))/((1-p)*(2-p))) - ((y*(mu**(1-p)))/(1-p)) - ((mu**(2-p))/(2-p)) )
    return dev    