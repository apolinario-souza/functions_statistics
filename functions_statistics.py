#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 09:11:33 2020

@author: tercio
"""

import pandas as pd 
import numpy  as np
from scipy.spatial import distance

#################### CRIANDO FUNCOES      ##################

#Criei uma função para criar os blocos
class estatistica():
    def media_bl(x,n_tt_bl,n_bl):
        y = np.zeros(n_bl)
        contador = 0;
        for i in range(n_bl):
             y[i] = np.mean(x[contador:contador+n_tt_bl])
             contador +=8;
        return y
    def desvio_bl(x,n_tt_bl,n_bl):
        y = np.zeros(n_bl)
        contador = 0;
        for i in range(n_bl):
             y[i] = np.std(x[contador:contador+n_tt_bl])
             contador +=8;
        return y
    
    #Criei de out
    
    def outlier(x):
        from scipy.stats import iqr
        y = x
        ref = iqr (y)*1.5;
        quat1 = np.quantile(y,0.25);
        quat3 = np.quantile(y,0.75);
        lim_cima = quat3 + ref;
        lim_baixo = quat1 - ref;
        for i in range (len(y)):
                if y[i] >= lim_cima or y[i] <= lim_baixo:
                        y[i] =  np.nan;
                else:
                        y[i] = y[i];
        m = np.nanmean (y);
        for k in range (len(y)):
            teste = np.isnan (y[k]); #Testa para ve se o valo tem nan, se tiver retorna 1
            if teste == 1:
               y[k] = m;
            else:
               y[k] = y[k]; 
               
        return y
#Approximate_entropy.https://en.wikipedia.org/wiki/Approximate_entropy
def ApEn(U, m, r):
    U = np.array(U)
    N = U.shape[0]
            
    def _phi(m):
        z = N - m + 1.0
        x = np.array([U[i:i+m] for i in range(int(z))])
        X = np.repeat(x[:, np.newaxis], 1, axis=2)
        C = np.sum(np.absolute(x - X).max(axis=2) <= r, axis=0) / z
        return np.log(C).sum() / z
    
    return abs(_phi(m + 1) - _phi(m))

def mahalanobis(data):
    
    maha = []
    for i in range((data.shape[0])-1):
        x,y = data[i,:], data[i+1,:]
        arr = np.array([x,y])
        cov = np.cov(arr.T)
        maha.append(distance.mahalanobis(x,y,cov))
    return np.mean(maha)

def make_log (data):
    data_new = np.zeros((data.shape)); 
    for k in range (data_new.shape[1]):
        for i in range(data_new.shape[0]):
            data_new[i,k] = np.log2 (data [i,k]) 
    
    return data_new


def make_log_1d (data):
    data_new = np.zeros((data.shape));
    for i in range(data_new.shape[0]):
        data_new[i] = np.log2 (data [i]) 
    
    return data_new


def transf_to_1_1d (data):
    data_new = np.zeros((data.shape));
    max = np.max(data)
    for i in range(data_new.shape[0]):
        data_new[i] = (data [i]*100)/max
    
    return data_new/100

def transf_to_1 (data):
    data_new = np.zeros((data.shape));
    
    for k in range (data_new.shape[1]):
        for i in range(data_new.shape[0]):
            max = np.max(data[:,k])
            data_new[i,k] = (data [i,k]*100)/max
    
    return data_new/100