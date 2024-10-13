from math import *
import random
import numpy as np
import pandas as pd
from collections import OrderedDict
from scipy.stats import norm, beta


class SimulatedSurvival_V:

    def __init__(self,T,step):
        self.T = T #100
        self.step = step #0.1
        self.t = np.arange(0,self.T,self.step)
            
    def g(self,s,w,x):
        return 0.01*(w+s*x)


    def generate_data(self, N, seed=0, test=False):

        np.random.seed(seed)
        t = self.t
        alpha_list = []
        Z_list = []
        W_list= []
        Y_list = [] #observation time
        delta_list = []
        cum_g_true_list = []
        hazard_true_list = []
        beta_1 = 1
        beta_0 = -3

        for i in range(N):
            alpha = np.random.uniform(0,1,5)
            Z = np.random.binomial(1,0.5,1)
            W = np.random.normal(0,1,1)
            alpha_list.append(alpha)
            Z_list.append(Z)
            W_list.append(W)
            X = alpha[0]+alpha[1]*np.sin(2*pi*t/self.T)+alpha[2]*np.cos(2*pi*t/self.T)+alpha[3]*np.sin(4*pi*t/self.T)+alpha[4]*np.cos(4*pi*t/self.T)
            g_x = [self.g(s,W,X[index]) for index,s in enumerate(t)]
            cum_g = np.cumsum(g_x)*self.step
            h = np.exp(beta_0+beta_1*Z+cum_g)
            ch = np.cumsum(h)*self.step
            S = np.exp(-ch)
            cum_g_true_list.append(cum_g[1:])
            hazard_true_list.append(h[1:])
            u = np.random.uniform(0,1,1)[0]
            index = np.sum(s>u for s in S) #S[index-1]>u, S[index]<=u
            if index==len(t):
                fT = t[-1]
            elif index==0:
                fT = 0.0001
            else:
                fT = t[index-1]+self.step*(S[index-1]-u)/(S[index-1]-S[index])
            C = min(np.random.exponential(50,1)[0],self.T-1) # change to adjust censoring rate
            Y_list.append(min(fT,C)) #observation time
            delta_list.append((fT<C)*1)

 
        print('Failue rate:', sum(delta_list)/len(delta_list))
        
        if test:
            partition_t = t[1:] # fixed partition
        else:
            partition_t = np.sort(Y_list)
            #partition_t = np.arange(0,self.T,0.5)
        
        p_n = len(partition_t)

        X = np.zeros((N,p_n))
        y = np.zeros((N,p_n))
        loc = np.zeros((N,p_n))
        ind = np.zeros((N,p_n))
        W = np.array(W_list)
        Z = np.array(Z_list)

        for i in range(N):
            Y = Y_list[i]
            delta = delta_list[i]
            alpha = alpha_list[i]
            for j, time in enumerate(partition_t):
                X[i][j] = alpha[0]+alpha[1]*np.sin(2*pi*time/self.T)+alpha[2]*np.cos(2*pi*time/self.T)+alpha[3]*np.sin(4*pi*time/self.T)+alpha[4]*np.cos(4*pi*time/self.T)
                #X[i][j] = alpha[0]
                loc[i][j] = time         
                if time < Y:
                    y[i][j] = 0.0
                elif time >= Y:
                    y[i][j] = delta
                if time <= Y:
                    ind[i][j] = 1.0
                else:
                    ind[i][j] = 0.0

        dt = np.diff(partition_t,prepend=0)
        loc = loc/100
        W = np.repeat(W,X.shape[-1],axis=1)
        #X = np.stack((X,loc,W),-1)
        Z = np.repeat(Z,X.shape[-1],axis=1)
        dt  = dt[np.newaxis,:,np.newaxis]
        dt = np.repeat(dt, N, axis=0)
        fail = y[:,:,np.newaxis]
        ind = ind[:,:,np.newaxis]
        if not test:
            return X,loc,W,Z,dt,fail,ind
        else:
            return X,loc,W,Z,dt,fail,ind,self.t[1:],cum_g_true_list,hazard_true_list
        
