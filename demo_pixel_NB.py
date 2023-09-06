# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 13:02:32 2023

@author: weiga
"""

##### load modules...
from __future__ import division
import sys
sys.path.insert(0, '/home/gw2397/cluster_new')
sys.path.insert(0, '/home/gw2397/pyhsmm-autoregressive-master')
sys.path.insert(0, '/home/gw2397/nbRegg_mcmc')

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import *
from scipy.interpolate import *
from cluster_functions import *
import scipy

import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov

import autoregressive.models as m
import autoregressive.distributions as d

from polyagamma import random_polyagamma
import numpy as np
from pyhmc import hmc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import gamma, digamma
from scipy.stats import norm, nbinom, multivariate_normal
from tqdm import tqdm
import heapq

import numpy.matlib
import time
from IPython import display

import warnings
import copy
warnings.filterwarnings("ignore")

######## read data
y = np.loadtxt("Y.csv", delimiter=",", dtype = int)

######## settings before MCMC
N = y.shape[0]
T = y.shape[1]
p = 2

n_iter = 1000
t_max = int(N/2)
state_max = 20

## neuron-clustering realted
sigma_alpha = 0.1 # scale for MH proposals in alpha move
alphaDP = 1

# MFM:
DPMM = False
alpha_random = False
MFMgamma = 1
# K ~ Geometric(r)
r = 0.415
log_pk = lambda k: np.log(r) + (k-1)*np.log(1-r)
a = MFMgamma
b = MFMgamma
log_v = MFMcoeff(log_pk, MFMgamma, N, t_max + 1)
logNb = np.log(np.arange(1,N+1) + b)

## state-clustering related
alpha_slds = 2.
gamma_slds = 4.
kappa_slds = 10

## other priors
prior = {}
prior['x0'] = np.zeros((p+1,1))
prior['Q0'] = np.eye(p+1)
prior['muC0'] = np.zeros((p,))
prior['SigC0'] = np.eye(p)
prior['mud0'] = 0
prior['s2d0'] = 1

prior['bA0'] = np.hstack((np.eye(p+1), np.zeros((p+1, 1))))
prior['Lam0'] = np.eye(p+1+1)
prior['Nu0'] = p+1+2
prior['Sig0'] = np.eye(p+1)*1e-2
a0 = 1
h = 1


## pre-allocation
t_fit = np.zeros((n_iter,), dtype = int)
Z_fit = np.zeros((n_iter, N), dtype = int)
numClus_fit = np.zeros((n_iter, t_max + 3), dtype = int)
actList_fit = np.ones((n_iter, t_max + 3), dtype = int)*-99 # different from MATLAB, cannot use 0 here

delt_fit = np.zeros((n_iter, N))
C_fit = np.zeros((n_iter, N, p))
muX_fit = [ [np.zeros((p+1, T)) for _ in range(t_max)] for _ in range(n_iter)]
dynamics_fit = []
for _ in range(n_iter):
    dynm = {}
    dynm['As'] = np.zeros((state_max,t_max*(p+1),t_max*(p+1)))
    dynm['bs'] = np.zeros((state_max,t_max*(p+1)))
    dynm['Qs'] = np.zeros((state_max,t_max*(p+1),t_max*(p+1)))
    dynm['states'] = np.zeros((T,), dtype = int)
    
    dynamics_fit.append(dynm)
    
r_fit = np.ones((n_iter, N))


## initialization

# start from 1 cluster
t_fit[0] = 1
Z_fit[0,:] = np.zeros((N,),dtype = int)
numClus_fit[0,0] = N
actList_fit[0,0] = 0
c_next = 1

delt_fit[0,:] = np.random.normal(prior['mud0'], prior['s2d0'], size = N)
C_fit[0,:] = np.random.multivariate_normal(prior['muC0'], prior['SigC0'], size = N)

for ii in range(n_iter):
    for qq in range(state_max):
        dynamics_fit[ii]['As'][qq,:,:] = np.eye(t_max*(p+1))
        dynamics_fit[ii]['Qs'][qq,:,:] = np.eye(t_max*(p+1))*1e-2

for jj in range(t_max):
    muX_fit[0][jj] = sample_muX(p,T)
    
muX_fit[0], C_fit[0,:], delt_fit[0,:], dynamics_fit[0] = constraint(muX_fit[0], C_fit[0,:,:],
                                                                    delt_fit[0,:].reshape((-1, 1)),
                                                                    dynamics_fit[0], Z_fit[0,:])

splitMerge_flag = False
splitL = 1
nSplit = 5
nMerge = 5

dynamics_sm = {}
dynamics_sm['As'] = np.zeros((state_max,p+1,p+1))
dynamics_sm['bs'] = np.zeros((state_max,p+1))
dynamics_sm['Qs'] = np.zeros((state_max,p+1,p+1))
dynamics_sm['states'] = np.zeros((T,), dtype = int)
dynamics_sm['As'][0,:,:] = np.eye(p+1)
dynamics_sm['Qs'][0,:,:] = np.eye(p+1)*1e-2

##### MCMC

keep_flg = False
fix_rep = 5
state_rep = 1

for gg in tqdm(range(1,n_iter)):
    
    
    #### 0. store previous steps
    muX_tmp = copy.deepcopy(muX_fit[gg-1])
    delt_tmp = copy.deepcopy(delt_fit[gg-1,:].reshape((-1, 1)))
    delt_tmp_raw = copy.deepcopy(delt_fit[gg-1,:])
    C_tmp = copy.deepcopy(C_fit[gg-1,:,:])
    r_tmp = copy.deepcopy(r_fit[gg-1,:].reshape((-1, 1)))
    r_tmp_raw = copy.deepcopy(r_fit[gg-1,:])
    dynamics_tmp = copy.deepcopy(dynamics_fit[gg-1])
    Z_tmp = copy.deepcopy(Z_fit[gg-1,:])
    
    for _ in range(fix_rep):
        #### 1. sample dynamical latents
        ## 1a. select corresponding muX & expand C to block diag
        lab_n_unique = sorted(set(Z_tmp))
        n_lab_n = len(lab_n_unique)
        lab_n_unique_dic = {}
        for kk in range(n_lab_n):
            lab_n_unique_dic[lab_n_unique[kk]] = kk

        muX_sel = np.zeros((n_lab_n*(p+1), T))

        As_sel = np.zeros((state_max,n_lab_n*(p+1),n_lab_n*(p+1)))
        bs_sel = np.zeros((state_max, n_lab_n*(p+1)))
        Qs_sel = np.zeros((state_max,n_lab_n*(p+1),n_lab_n*(p+1)))
        C_trans = np.zeros((N, n_lab_n*(p+1)))

        sel_idx = np.array([], dtype = int)
        for sn in lab_n_unique:
            idx_tmp = lab_n_unique_dic[sn]
            muX_sel[(idx_tmp*(p+1)):((idx_tmp+1)*(p+1)),:] = muX_tmp[sn]
            sel_idx = np.append(sel_idx, np.arange(sn*(p+1), (sn+1)*(p+1)))

        for st in range(state_max):
            As_sel[st,:,:] = dynamics_tmp['As'][st,:,:][np.ix_(sel_idx, sel_idx)]
            bs_sel[st,:] = dynamics_tmp['bs'][st,sel_idx]
            Qs_sel[st,:,:] = dynamics_tmp['Qs'][st,:,:][np.ix_(sel_idx, sel_idx)]


        for n_id, zz in enumerate(Z_tmp):
            idx_tmp = lab_n_unique_dic[zz]
            C_trans[n_id, (idx_tmp*(p+1)):((idx_tmp+1)*(p+1))] = np.hstack((1, C_tmp[n_id, :]))

        ## 1b. sample mu & X

        # 1b.01 PG augmentation (transform Y)
        omega = random_polyagamma(r_tmp + y, delt_tmp + C_trans @ muX_sel - np.log(r_tmp))
        kappa = (y-r_tmp)/2 + omega*(np.log(r_tmp) - delt_tmp)
        y_hat = (1/omega)*kappa

        # 1b.02 FFBS
        x0_tmp = np.repeat(prior['x0'], n_lab_n).reshape((-1, 1))
        Q0_tmp = np.kron(np.eye(n_lab_n,dtype=int),prior['Q0'])
        muX_update = FFBS(y_hat,C_trans, 1/omega, dynamics_tmp['states'], As_sel, bs_sel, Qs_sel, x0_tmp, Q0_tmp)

        for sn in lab_n_unique:
            idx_tmp = lab_n_unique_dic[sn]
            muX_fit[gg][sn] = muX_update[(idx_tmp*(p+1)):((idx_tmp+1)*(p+1)),:]

        muX_fit[gg], C_tmp, delt_tmp_raw, dynamics_tmp = constraint(muX_fit[gg], C_tmp,delt_tmp,
                                                                    dynamics_tmp, Z_tmp, muX_tmp)
        delt_tmp = delt_tmp_raw.reshape((-1,1))
        
        for sn in lab_n_unique:
            idx_tmp = lab_n_unique_dic[sn]
            muX_update[(idx_tmp*(p+1)):((idx_tmp+1)*(p+1)),:] = muX_fit[gg][sn]


        ## 1c. sample states & dynamics (b, A & Q)

        Nu0_tmp = n_lab_n*(p+1)+2
        Sig0_tmp = np.kron(np.eye(n_lab_n,dtype=int),prior['Sig0'])
        bA0_tmp = np.hstack((np.kron(np.eye(n_lab_n,dtype=int),prior['bA0'][:,0:-1]), np.repeat(prior['bA0'][:,-1], n_lab_n).reshape((-1, 1))))
        Lam0_tmp = np.eye(n_lab_n*(p+1)+1)

        model = m.ARWeakLimitStickyHDPHMM(
            alpha=alpha_slds, gamma=gamma_slds, kappa=kappa_slds, 
            init_state_distn='uniform',
            obs_distns=[d.AutoRegression(nu_0=Nu0_tmp,S_0=Sig0_tmp,
                                         M_0=bA0_tmp,K_0=Lam0_tmp,affine=True) for state in range(state_max)],)
        model.add_data(muX_update.T)

        if (gg > 0.05*n_iter and max([np.sum(dynamics_tmp['states'] == s) for s in np.unique(dynamics_tmp['states'])]) > 0.8*T):
            keep_flg = False
            print('no keep')

        if keep_flg:
            model.states_list[0].stateseq = dynamics_tmp['states'][0:-1].astype(np.int32).copy()
            print('keep')
        else:
            if (gg > 0.05*n_iter and max([np.sum(dynamics_tmp['states'] == s) for s in np.unique(dynamics_tmp['states'])]) < 0.7*T):
                model.states_list[0].stateseq = dynamics_tmp['states'][0:-1].astype(np.int32).copy()
                keep_flg = True
                print('keep')
            else:
                print('no keep')

        for _ in range(state_rep):
            model.resample_model()


        ## 1d. allocation back...

        # states
        dynamics_fit[gg]['states'][0:-1] = model.states_list[0].stateseq.copy()
        dynamics_fit[gg]['states'][-1] = dynamics_fit[gg]['states'][-2].copy()

        # muX
        for sn in lab_n_unique:
            idx_tmp = lab_n_unique_dic[sn]
            muX_fit[gg][sn] = muX_update[(idx_tmp*(p+1)):((idx_tmp+1)*(p+1)),:]

        # linear dynamics (b, A, Q): this redundant, only assign used states later...
        # no time to do it now...
        for idx, s in enumerate(model.obs_distns):
            dynamics_fit[gg]['bs'][idx,sel_idx] = s.A[:,-1]
            dynamics_fit[gg]['As'][idx,:,:][np.ix_(sel_idx, sel_idx)] = s.A[:,:-1]
            dynamics_fit[gg]['Qs'][idx,:,:][np.ix_(sel_idx, sel_idx)] = s.sigma



        #### 2. sample loadings (delt, C)
        delt_b, C_b = update_deltC(delt_tmp, C_tmp, y,
                                   r_tmp_raw.ravel(), Z_tmp.ravel(), muX_fit[gg], prior)
        delt_fit[gg,:] = delt_b.ravel()
        C_fit[gg,:,:] = C_b

        #### 3. sample r
        delt_tmp = delt_fit[gg,:].reshape((-1, 1))
        C_trans = np.zeros((N, n_lab_n*(p+1)))
        for n_id, zz in enumerate(Z_tmp):
            idx_tmp = lab_n_unique_dic[zz]
            C_trans[n_id, (idx_tmp*(p+1)):((idx_tmp+1)*(p+1))] = np.hstack((1, C_fit[gg,n_id,:]))
        Mu_r = np.exp(delt_tmp + C_trans @ muX_update)

        try: 
            r_b = update_r(y,Mu_r,r_tmp_raw, a0=a0, h=h, use_hmc=False)
        except: 
            r_b = update_r(y,Mu_r,r_tmp_raw, a0=a0, h=h, use_hmc=True)


        r_fit[gg,:] = copy.deepcopy(r_b)
        r_fit[gg,r_fit[gg,:] > 100] = copy.deepcopy(r_tmp_raw[r_fit[gg,:] > 100])
        
        
        #### 4. projection    
        muX_fit[gg], C_fit[gg,:,:], delt_fit[gg,:], dynamics_fit[gg] = constraint(muX_fit[gg], C_fit[gg,:,:],
                                                                                  delt_fit[gg,:].reshape((-1, 1)),
                                                                                  dynamics_fit[gg], Z_tmp, muX_tmp)
        
        muX_tmp = copy.deepcopy(muX_fit[gg])
        delt_tmp = copy.deepcopy(delt_fit[gg,:].reshape((-1, 1)))
        delt_tmp_raw = copy.deepcopy(delt_fit[gg,:])
        C_tmp = copy.deepcopy(C_fit[gg,:,:])
        r_tmp = copy.deepcopy(r_fit[gg,:].reshape((-1, 1)))
        r_tmp_raw = copy.deepcopy(r_fit[gg,:])
        dynamics_tmp = copy.deepcopy(dynamics_fit[gg])
        
    muX_tmp, C_tmp, dynamics_tmp, M1_tmp = contraint_ortho(muX_fit[gg], C_fit[gg,:,:], dynamics_fit[gg], Z_fit[gg-1,:])

    #### 5. clustering...
    # 5a. use split-merge...
    if splitMerge_flag and (gg % 20 == 1):

        Z_fit[gg-1,:], numClus_fit[gg-1,:], t_fit[gg-1], actList_fit[gg-1,:], muX_tmp, delt_fit[gg,:], C_tmp, dynamics_tmp= \
        splitMerge(Z_fit[gg-1,:], numClus_fit[gg-1,:], t_fit[gg-1], actList_fit[gg-1,:],
                   muX_tmp,delt_fit[gg,:].reshape((-1, 1)),C_tmp,y,r_fit[gg,:],dynamics_sm,prior,
                   a, b, log_v,np.random.binomial(1, .5), nSplit, nMerge,dynamics_tmp)

        c_next = ordered_next(actList_fit[gg-1,:])


    # 5b. regular
    if gg % 10 == 1: sample_tag = 1
    else: sample_tag = 0

    Z_fit[gg,:], numClus_fit[gg,:], t_fit[gg], actList_fit[gg,:], c_next, muX_tmp, dynamics_tmp = \
    update_cluster(Z_fit[gg-1,:], numClus_fit[gg-1,:], t_fit[gg-1], actList_fit[gg-1,:],
                   c_next,DPMM, alpha_random,
                   a, log_v, logNb, muX_tmp, delt_fit[gg,:].reshape((-1, 1)),
                   y, r_fit[gg,:], prior, alphaDP,sigma_alpha,t_max,dynamics_tmp, sample_tag)
    
    # do more for newly assigned
    Z_more = np.setdiff1d(np.unique(Z_fit[gg,:]), np.unique(Z_fit[gg-1,:]))
    
    for _ in range(5):
        for zz in Z_more:
            obsIdx = Z_fit[gg,:] == zz
            delt_tmp = delt_fit[gg,obsIdx].reshape((-1, 1))
            C_tmp = C_fit[gg,obsIdx,:]
            y_tmp = y[obsIdx,:]
            r_tmp = r_fit[gg,obsIdx].reshape((-1, 1))

            C_expand = np.column_stack((np.ones((C_tmp.shape[0],)), C_tmp))
            # 1) PG-augmentation
            omega = random_polyagamma(r_tmp + y_tmp, delt_tmp + C_expand @ muX_fit[gg][zz] - np.log(r_tmp))
            kappa = (y_tmp-r_tmp)/2 + omega*(np.log(r_tmp) - delt_tmp)
            y_hat = (1/omega)*kappa

            # 2) FFBS
            muX_b = FFBS(y_hat,C_expand,1/omega,dynamics_sm['states'], dynamics_sm['As'],
                         dynamics_sm['bs'],dynamics_sm['Qs'],prior['x0'],prior['Q0'])
            muX_fit[gg][zz] = copy.deepcopy(muX_b)
            
            
            delt_b, C_b = update_deltC(delt_fit[gg,obsIdx].reshape((-1, 1)), C_fit[gg,obsIdx,:], y_tmp,
                                r_fit[gg,obsIdx].ravel(), Z_fit[gg,obsIdx].ravel(), muX_fit[gg], prior)
            delt_fit[gg,obsIdx] = copy.deepcopy(delt_b.ravel())
            C_fit[gg,obsIdx,:] = copy.deepcopy(C_b)
            
    muX_fit[gg], C_fit[gg,:,:], delt_fit[gg,:], dynamics_fit[gg] = constraint(muX_fit[gg], C_fit[gg,:,:],
                                                                              delt_fit[gg,:].reshape((-1, 1)),
                                                                              dynamics_fit[gg], Z_fit[gg,:], muX_fit[gg]) 
    
    #### plotting...

############### store results...
states_fit = np.zeros((n_iter, T))
for gg in range(n_iter):
    states_fit[gg,:] = dynamics_fit[gg]['states']

import pickle

## write...
with open('new_NB_rep_20_55_c1_t_fit.pkl', 'wb') as f: pickle.dump(t_fit, f)
with open('new_NB_rep_20_55_c1_Z_fit.pkl', 'wb') as f: pickle.dump(Z_fit, f)
with open('new_NB_rep_20_55_c1_states_fit.pkl', 'wb') as f: pickle.dump(states_fit, f)
with open('new_NB_rep_20_55_c1_muX_fit.pkl', 'wb') as f: pickle.dump(muX_fit, f)













