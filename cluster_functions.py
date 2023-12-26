# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:44:26 2023

@author: weiga
"""
######################################################
#### import modules 
from __future__ import division

import sys
sys.path.insert(0, '/home/gw2397/cluster_new')
sys.path.insert(0, '/home/gw2397/pyhsmm-autoregressive-master')
sys.path.insert(0, '/home/gw2397/nbRegg_mcmc')

import copy
import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import *
from scipy.interpolate import *

import pyhsmm

import autoregressive.models as m
import autoregressive.distributions as d

from polyagamma import random_polyagamma
from pyhmc import hmc
import statsmodels.api as sm
from scipy.special import gamma, digamma
from scipy.stats import norm, nbinom, multivariate_normal
import heapq

import warnings
warnings.filterwarnings("ignore")

######################################################
##### utils

lAbsGam = lambda x: np.log(np.abs(gamma(x)))

def random_rotation(n, theta=None):
    if theta is None:
        # Sample a random, slow rotation
        theta = 0.5 * np.pi * np.random.rand()

    if n == 1:
        return np.random.rand() * np.eye(1)

    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    out = np.eye(n)
    out[:2, :2] = rot
    q = np.linalg.qr(np.random.randn(n, n))[0]
    return q.dot(out).dot(q.T)

def jitters_muX(muX_tmp):
    muX = muX_tmp + np.random.normal(size = muX_tmp.shape)*0.1
    return muX


def sample_muX(p,T):
    muX = np.ones((p+1, T))*np.Inf
    for pp in range(p+1):
        while np.sum(np.abs(muX[pp,:]) > 2) > 1:
            k = np.ceil(np.random.uniform()*25).astype(int) + 10
            f = splrep(np.linspace(0,1,k), np.random.uniform(-.2,.2,size = k))
            muX[pp,:] = splev(np.linspace(0,1,T), f)
            muX[pp,:] = muX[pp,:] - np.mean(muX[pp,:])

#     U,_,_ = np.linalg.svd(muX[1:,:].T, full_matrices= False)
#     muX[1:,:] = U.T       
            
    return muX

def sample_muX2(p,T):
    
    As = [0.95*np.eye(p+1)]
    truemodel = m.ARHSMM(
        alpha=5.,init_state_concentration=5.,
        obs_distns=[d.AutoRegression(A=A,sigma=np.eye(p+1)*1e-2) for A in As],
        dur_distns=[pyhsmm.basic.distributions.PoissonDuration(alpha_0=100,beta_0=2)
            for state in range(len(As))],
        )
    X_all_tmp, lab_t_tmp = truemodel.generate(2*T, keep=True)
    muX = X_all_tmp[T+1:,:].T
    
    # constraint...
    muX = muX - np.mean(muX, 1).reshape((-1, 1))
#     U,_,_ = np.linalg.svd(muX[1:,:].T, full_matrices= False)
#     muX[1:,:] = U.T

    return muX

# def sample_muX3(p,T, dynamics_tmp, prior, clus_tmp):
    
#     # to debug
#     # dynamics_tmp = dynamics_fit[gg]
#     # clus_tmp = 0
    
#     muX = np.zeros((p+1, T))
#     # muX[:,0] = np.random.multivariate_normal(prior['x0'].ravel(), prior['Q0'])

#     for tt in range(1,T):
#         state_tmp = dynamics_tmp['states'][tt-1]
        
#         A_tmp = dynamics_tmp['As'][state_tmp][clus_tmp*(p+1):((clus_tmp+1)*(p+1)),
#                                               clus_tmp*(p+1):((clus_tmp+1)*(p+1))]
#         b_tmp = dynamics_tmp['bs'][state_tmp][clus_tmp*(p+1):((clus_tmp+1)*(p+1))]
#         Q_tmp = dynamics_tmp['Qs'][state_tmp][clus_tmp*(p+1):((clus_tmp+1)*(p+1)),
#                                               clus_tmp*(p+1):((clus_tmp+1)*(p+1))]
        
#         muX[:,tt] = (A_tmp @ muX[:,tt-1].reshape((-1, 1)) + b_tmp.reshape((-1, 1)) +
#                      np.random.multivariate_normal(np.zeros((p+1,)), Q_tmp).reshape((-1, 1))).ravel()
    
#     return muX

def constraint(muX_a, C_a, delt_a, dynamics_a, Z_tmp, muX_pre = None):
    
    ## I need to be careful about the label switching...
    ## easy solutions...
    ## 1) modify original labels, or (let's do this first...)
    ## 2) modify the matching labels
    
    # to debug...
    # muX_a = muX_fit[gg]
    # C_a = C_fit[gg,:,:]
    # delt_a = delt_fit[gg,:].reshape((-1, 1))
    # dynamics_a = dynamics_fit[gg]
    # Z_tmp = Z_fit[gg-1,:]
    # muX_pre = muX_fit[gg-1]
    
    muX_b = copy.deepcopy(muX_a)
    C_b = copy.deepcopy(C_a)
    delt_b = copy.deepcopy(delt_a)
    dynamics_b = copy.deepcopy(dynamics_a)
    lab_unique = np.unique(Z_tmp)
    
    lat_dim = dynamics_b['As'].shape[1]
    trans_all = np.zeros((lat_dim,1))
    rot_all = np.eye(lat_dim)
    
    p_tmp = C_a.shape[1]
    perm_idx = np.repeat(np.arange(0,p_tmp),2)
    sign_idx = np.tile([1, -1], p_tmp)
    idx_raw = np.arange(0, 2*p_tmp)
    
    for ll in lab_unique:
        
        muX_bar = np.mean(muX_b[ll], axis = 1).reshape((-1, 1))
        obsIdx = Z_tmp == ll
        n_tmp = np.sum(obsIdx)
        
        # (1) translation
        muX_b[ll] = muX_b[ll] - muX_bar
        delt_b[obsIdx] = delt_b[obsIdx] + np.hstack((np.ones((n_tmp,1)), C_b[obsIdx,:])) @ muX_bar
        trans_all[(ll*(p_tmp+1)):((ll+1)*(p_tmp+1))] = muX_bar
        
        # (2) rotation: be careful... (how to get the orthonormal matrix?)
        X_tmp = muX_b[ll][1:,:].copy()
        
        U, S, Vh = np.linalg.svd(X_tmp.T, full_matrices= False)
        US = U @ np.diag(S)
        
        if muX_pre is not None:
            X_tmp_pre = muX_pre[ll][1:,:].copy()
            min_indx = np.zeros((p_tmp,), dtype = int)
            idx_set = np.arange(0, 2*p_tmp)
            Xout = np.zeros_like(US.T)

            for rr in range(p_tmp):
                mse_tmp = np.zeros((2*p_tmp,))
                for cc in range(2*p_tmp):
                    mse_tmp[cc] = np.mean((X_tmp_pre[rr,:] - sign_idx[cc]*US[:,perm_idx[cc]])**2)
                sorted_ind = np.argsort(mse_tmp)
                min_indx[rr] = sorted_ind[np.isin(sorted_ind, idx_set)][0]
                Xout[rr,:] = sign_idx[min_indx[rr]]*US[:,perm_idx[min_indx[rr]]]
                idx_set = idx_set[~np.isin(idx_set, idx_raw[perm_idx == perm_idx[min_indx[rr]]])]

            # sign
            Vh = Vh[perm_idx[min_indx],:]
            Vh = Vh @ np.diag(sign_idx[min_indx])
            
        else:
            Xout = US.T
        M_dag = Vh
        
        # np.allclose(M_dag @ X_tmp, Xout)
        
        
        muX_b[ll][1:,:] = Xout
        
        rot_all[(ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1)), (ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1))] = M_dag
        C_b[obsIdx,:] = C_b[obsIdx,:] @ np.linalg.inv(M_dag)
        
    
    # for linear dyanmics (A, b, Q)...
    for mm in range(dynamics_b['As'].shape[0]):
        
        # (1) translation
        dynamics_b['bs'][mm,:] = ((dynamics_b['As'][mm,:,:] - np.eye(lat_dim)) @ trans_all).ravel() + dynamics_b['bs'][mm,:]
        
        # (2) rotation
        dynamics_b['As'][mm,:,:] = rot_all @ dynamics_b['As'][mm,:,:] @ np.linalg.inv(rot_all)
        dynamics_b['bs'][mm,:] = (rot_all @ dynamics_b['bs'][mm,:].reshape((-1, 1))).ravel()
        dynamics_b['Qs'][mm,:,:] = rot_all @ dynamics_b['Qs'][mm,:,:] @ rot_all.T
    
    return muX_b, C_b, delt_b.ravel(), dynamics_b


def contraint_ortho(muX_a, C_a, dynamics_a, Z_tmp):
    
    # to debug
    # muX_a = muX_tmp
    # C_a = C_tmp
    # dynamics_a = dynamics_tmp
    # Z_tmp = Z_fit[gg,:]
    
    muX_b = copy.deepcopy(muX_a)
    C_b = copy.deepcopy(C_a)
    dynamics_b = copy.deepcopy(dynamics_a)
    lab_unique = np.unique(Z_tmp)

    lat_dim = dynamics_b['As'].shape[1]
    rot_all = np.eye(lat_dim)
    p_tmp = C_a.shape[1]

    for ll in lab_unique:
        X_tmp = muX_b[ll][1:,:].copy()
        obsIdx = Z_tmp == ll
        U, S, Vh = np.linalg.svd(X_tmp.T, full_matrices= False)

        Xout = U.T
        M_dag = np.diag(1/S) @ Vh

        muX_b[ll][1:,:] = Xout
        rot_all[(ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1)), (ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1))] = M_dag
        C_b[obsIdx,:] = C_b[obsIdx,:] @ np.linalg.inv(M_dag)

    for mm in range(dynamics_b['As'].shape[0]):

        # (2) rotation
        dynamics_b['As'][mm,:,:] = rot_all @ dynamics_b['As'][mm,:,:] @ np.linalg.inv(rot_all)
        dynamics_b['bs'][mm,:] = (rot_all @ dynamics_b['bs'][mm,:].reshape((-1, 1))).ravel()
        dynamics_b['Qs'][mm,:,:] = rot_all @ dynamics_b['Qs'][mm,:,:] @ rot_all.T 
        
    return  muX_b, C_b, dynamics_b, rot_all

def contraint_back(C_tmp, muX_tmp, dynamics_tmp, M_tmp, Z_tmp):
    
    #     M_tmp = M2_tmp @ M1_tmp
    #     Z_tmp = Z_fit[gg-1,:]

    M_tmp_inv = np.linalg.inv(M_tmp)
    C_b = copy.deepcopy(C_tmp)
    muX_b = copy.deepcopy(muX_tmp)
    dynamics_b = copy.deepcopy(dynamics_tmp)
    lab_unique = np.unique(Z_tmp)

    p_tmp = C_tmp.shape[1]

    for ll in lab_unique:

        obsIdx = Z_tmp == ll
        M_dag = M_tmp[(ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1)), (ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1))]
        muX_b[ll][1:,:] = np.linalg.inv(M_dag) @ muX_tmp[ll][1:,:]
        C_b[obsIdx,:] = C_tmp[obsIdx,:] @ M_dag


    for mm in range(dynamics_b['As'].shape[0]):

        # (2) rotation
        dynamics_b['As'][mm,:,:] = M_tmp_inv @ dynamics_b['As'][mm,:,:] @ M_tmp
        dynamics_b['bs'][mm,:] = (M_tmp_inv @ dynamics_b['bs'][mm,:].reshape((-1, 1))).ravel()
        dynamics_b['Qs'][mm,:,:] = M_tmp_inv @ dynamics_b['Qs'][mm,:,:] @ M_tmp_inv.T 

    return C_b, muX_b, dynamics_b
    


# def constraint(muX_a, C_a, delt_a, dynamics_a, Z_tmp, muX_pre = None):
    
#     # to debug...
#     # muX_a = muX_fit[gg]
#     # C_a = C_fit[gg,:,:]
#     # delt_a = delt_fit[gg,:].reshape((-1, 1))
#     # dynamics_a = dynamics_fit[gg]
#     # Z_tmp = Z_fit[gg-1,:]
#     # muX_pre = muX_fit[gg-1]
    
#     muX_b = muX_a.copy()
#     C_b = C_a.copy()
#     delt_b = delt_a.copy()
#     dynamics_b = dynamics_a.copy()
#     lab_unique = np.unique(Z_tmp)
    
#     lat_dim = dynamics_b['As'].shape[1]
#     trans_all = np.zeros((lat_dim,1))
#     rot_all = np.eye(lat_dim)
    
#     p_tmp = C_a.shape[1]
#     perm_idx = np.repeat(np.arange(0,p_tmp),2)
#     sign_idx = np.tile([1, -1], p_tmp)
#     idx_raw = np.arange(0, 2*p_tmp)
    
#     for ll in lab_unique:
        
#         muX_bar = np.mean(muX_b[ll], axis = 1).reshape((-1, 1))
#         obsIdx = Z_tmp == ll
#         n_tmp = np.sum(obsIdx)
        
#         # (1) translation
#         muX_b[ll] = muX_b[ll] - muX_bar
#         delt_b[obsIdx] = delt_b[obsIdx] + np.hstack((np.ones((n_tmp,1)), C_b[obsIdx,:])) @ muX_bar
#         trans_all[(ll*(p_tmp+1)):((ll+1)*(p_tmp+1))] = muX_bar
        
#         # (2) rotation: be careful... (how to get the orthonormal matrix?)
#         X_tmp = muX_b[ll][1:,:].copy()
        
        
#         # Q = gram_schmidt(X_tmp.T)
#         # R = Q.T @ X_tmp.T
#         # muX_b[ll][1:,:] = Q.T
#         # M_dag = R.T
#         # np.allclose(M_dag @ X_tmp, Q.T)
        
#         U, S, Vh = np.linalg.svd(X_tmp.T, full_matrices= False)
        
#         if muX_pre is not None:
#             X_tmp_pre = muX_pre[ll][1:,:].copy()
#             min_indx = np.zeros((p_tmp,), dtype = int)
#             idx_set = np.arange(0, 2*p_tmp)
#             Xout = np.zeros_like(U.T)

#             for rr in range(p_tmp):
#                 mse_tmp = np.zeros((2*p_tmp,))
#                 for cc in range(2*p_tmp):
#                     mse_tmp[cc] = np.mean((X_tmp_pre[rr,:] - sign_idx[cc]*U[:,perm_idx[cc]])**2)
#                 sorted_ind = np.argsort(mse_tmp)
#                 min_indx[rr] = sorted_ind[np.isin(sorted_ind, idx_set)][0]
#                 Xout[rr,:] = sign_idx[min_indx[rr]]*U[:,perm_idx[min_indx[rr]]]
#                 idx_set = idx_set[~np.isin(idx_set, idx_raw[perm_idx == perm_idx[min_indx[rr]]])]

#             # sign
#             S = S * sign_idx[min_indx]
#             # permute
#             S = S[perm_idx[min_indx]]
#             Vh = Vh[perm_idx[min_indx],:]
#         else:
#             Xout = U.T
#         M_dag = np.diag(1/S) @ Vh
        
# #         np.allclose(Xout.T @ np.diag(S) @ Vh, X_tmp.T)
# #         np.allclose(M_dag @ X_tmp, Xout)
        
#         muX_b[ll][1:,:] = Xout
        
#         ### S descending order > 0 & 1st entry of U is >0 (pretty worried about this...)
#         # (1) descentding order & > 0
# #         np.allclose(U @ np.diag(S) @ Vh, X_tmp.T)
        
# #         sign1 = np.diag(-1+2*(U[0,:] > 0))
# #         sign2 = np.diag(-1 + 2*(S>0))
# #         perm = np.argsort(-np.abs(S))
# #         U = (U @ sign1)[:,perm]
# #         S = np.abs(S[perm])
# #         Vh = (sign1 @ sign2 @ Vh)[perm,:]
        
# #         M_dag = np.diag(1/S) @ Vh
# #         muX_b[ll][1:,:] = U.T
        
#         rot_all[(ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1)), (ll*(p_tmp+1)+1):((ll+1)*(p_tmp+1))] = M_dag
#         C_b[obsIdx,:] = C_b[obsIdx,:] @ np.linalg.inv(M_dag)
        
    
#     # for linear dyanmics (A, b, Q)...
#     for mm in range(dynamics_b['As'].shape[0]):
        
#         # (1) translation
#         dynamics_b['bs'][mm,:] = ((dynamics_b['As'][mm,:,:] - np.eye(lat_dim)) @ trans_all).ravel() + dynamics_b['bs'][mm,:]
        
#         # (2) rotation
#         dynamics_b['As'][mm,:,:] = rot_all @ dynamics_b['As'][mm,:,:] @ np.linalg.inv(rot_all)
#         dynamics_b['bs'][mm,:] = (rot_all @ dynamics_b['bs'][mm,:].reshape((-1, 1))).ravel()
#         dynamics_b['Qs'][mm,:,:] = rot_all @ dynamics_b['Qs'][mm,:,:] @ rot_all.T
    
#     return muX_b, C_b, delt_b.ravel(), dynamics_b



######################################################
##### update linear dynamics

def FFBS(Y_ffbs,X_ffbs,SigFlat_ffbs,states_ffbs, As_ffbs,bs_ffbs,Qs_ffbs,m0_ffbs,V0_ffbs):
    
    T = Y_ffbs.shape[1]
    k = X_ffbs.shape[1]
    
    m_tmp = np.zeros((k,T))
    V_tmp = np.zeros((T,k,k))
    
    # step 1: FF: forward filtering
    for tt in range(T):

        state_tmp = states_ffbs[tt]
        A_tmp = As_ffbs[state_tmp,:,:]
        b_tmp = bs_ffbs[state_tmp,:]
        Q_tmp = Qs_ffbs[state_tmp,:,:]

        if tt == 0:
            m_tt_1 = A_tmp @ m0_ffbs + b_tmp.reshape(-1, 1)
            V_tt_1 = A_tmp @ V0_ffbs @ A_tmp.T + Q_tmp
        else:
            m_tt_1 = A_tmp @ m_tmp[:,tt-1].reshape(-1, 1) + b_tmp.reshape(-1, 1)
            V_tt_1 = A_tmp @ V_tmp[tt-1,:,:] @ A_tmp.T + Q_tmp

        obs_idx = ~np.isnan(Y_ffbs[:,tt])

        X_tmp = X_ffbs[obs_idx,:]
        Sig_tmp = np.diag(SigFlat_ffbs[obs_idx,tt])
        Y_tmp = Y_ffbs[obs_idx,tt].reshape(-1, 1)

        Kt = V_tt_1 @ X_tmp.T @ np.linalg.inv(X_tmp @ V_tt_1 @ X_tmp.T + Sig_tmp)
        m_tmp[:,tt] = (m_tt_1 + Kt @ (Y_tmp - X_tmp @ m_tt_1)).ravel()
        V_tmp[tt,:,:] = (np.eye(k) - Kt @ X_tmp) @ V_tt_1
        V_tmp[tt,:,:] = (V_tmp[tt,:,:] + V_tmp[tt,:,:].T)/2
    
    
    # step 2: BS: backward smapling
    BETA_out = np.zeros((k,T))
    BETA_out[:,T-1] = np.random.multivariate_normal(m_tmp[:,T-1], V_tmp[T-1,:,:])
    
    for tt in reversed(range(T-1)):
        
        state_tmp = states_ffbs[tt]
        
        A_tmp = As_ffbs[state_tmp,:,:]
        b_tmp = bs_ffbs[state_tmp,:]
        Q_tmp = Qs_ffbs[state_tmp,:,:]
        
        Jt = V_tmp[tt,:,:] @ A_tmp.T @ np.linalg.inv(A_tmp @ V_tmp[tt,:,:] @ A_tmp.T + Q_tmp)
        mstar_tmp = m_tmp[:,tt].reshape(-1, 1) + Jt @ (BETA_out[:,tt+1].reshape(-1, 1) -
                                                       A_tmp @ m_tmp[:,tt].reshape(-1, 1) -
                                                       b_tmp.reshape(-1, 1))
        Vstar_tmp = (np.eye(k) - Jt @ A_tmp) @ V_tmp[tt,:,:]
        Vstar_tmp = (Vstar_tmp + Vstar_tmp.T)/2
        BETA_out[:,tt] = np.random.multivariate_normal(mstar_tmp.ravel(), Vstar_tmp)
    
    return BETA_out



######################################################
##### update loading & bias

def update_deltC(delt_a, C_a, Y_tmp, r_tmp, Z_tmp, muX_tmp, prior):
    
    # to debug
    # delt_a = dd.copy()
    # C_a = C_true.copy()
    # Y_tmp = y
    # r_tmp = r_true.ravel()
    # Z_tmp = lab_neuron
    # muX_tmp = muX_all
    # prior
    
    N = Y_tmp.shape[0]
    T = Y_tmp.shape[1]
    
    delt_b = copy.deepcopy(delt_a)
    C_b = copy.deepcopy(C_a)
    
    prior_b = np.append(prior['mud0'], prior['muC0'])
    prior_B = block_diag(prior['s2d0'], prior['SigC0'])
    
    for ii in range(N):
        
        # 1. Sample PG variables:
        # omega ~ PG(y+r, delta + x'\beta - log{r})
        # Kappa = (y-r)/2 + omega*(log{r} - delta)
        # then the transformed \hat{y} = omega^{-1}\kappa \sim N(X\beta, \omega^{-1})
        
        y_ii = Y_tmp[ii,:]
        r_ii = r_tmp[ii]
        
        off_ii = muX_tmp[Z_tmp[ii]][0,:]
        X_ii = np.hstack((np.ones((T,1)),muX_tmp[Z_tmp[ii]][1:,:].T))
        beta_ii = np.hstack((delt_b[ii], C_b[ii,:]))[:, None]
        
        omega = random_polyagamma(y_ii + r_ii, off_ii.ravel() + (X_ii @ beta_ii).ravel() - np.log(r_ii))
        Kappa = (y_ii - r_ii)/2 + omega*(np.log(r_ii)- off_ii.ravel())
        Omega = np.diag(omega)
        
        # 2. update delta & C:
        # \beta|- \sim N(m, V)
        # V = (X'\Omega X + B^{-1})^{-1}
        # m = V(X'\Kappa + B^{-1}*b)
        
        V_inv = X_ii.T @ Omega @ X_ii + np.linalg.inv(prior_B)
        V = np.linalg.inv(V_inv)
        m = V @ (X_ii.T @ Kappa.reshape((-1, 1)) + np.linalg.inv(prior_B) @ prior_b.reshape((-1, 1)))
        samp_ii = np.random.multivariate_normal(m.ravel(),V)
        
        delt_b[ii] = samp_ii[0]
        C_b[ii,:] = samp_ii[1:]
    
    return delt_b, C_b

######################################################
##### update dispersion (r)

def CRT_sum(x,r):
    Lsum = 0
    RND = r/(r+np.arange(0,np.max(x)))
    for ii in range(x.size):
        if x[ii] > 0:
            Lsum = Lsum + np.sum(np.random.uniform(size = x[ii]) <= RND[0:x[ii]])
    return(Lsum)

def lg_pdf(r,pp,y,a0,h):
    llhd = np.sum(np.log(gamma(r+y.ravel())) - np.log(gamma(r)) + r*np.log(1-pp.ravel()))
    lprior = (a0-1)*np.log(r) - h*r
    logp = llhd + lprior

    llhd_grad = np.sum(digamma(r+y.ravel()) - digamma(r) + np.log(1-pp.ravel()))
    lprior_grad = (a0-1)/r - h
    grad = llhd_grad + lprior_grad

    return logp, grad

def update_r_single(y_rs,mu_rs,rPre_rs, a0, h, use_hmc):
    
    obsIdx = ~np.isnan(y_rs)

    if use_hmc:
        p_tmp = mu_rs[obsIdx]/(mu_rs[obsIdx] + rPre_rs)
        lg_pdf_tmp = lambda r: lg_pdf(r,p_tmp,y_rs[obsIdx],a0,h)
        samples = hmc(lg_pdf_tmp, x0=np.array([rPre_rs]), n_samples=1)
        r_out = samples
    
    else:
        Lsum = CRT_sum(y_rs[obsIdx].astype(int),rPre_rs)
        r_out = np.random.gamma(a0 + Lsum, 1/(h-np.sum(np.log(rPre_rs/(rPre_rs + mu_rs[obsIdx])))))

    return r_out

def update_r(y_r,Mu_r,R_pre_r, a0=1, h=1, use_hmc=False):
    
    N = Mu_r.shape[0]
    R_out = np.ones(N)
    
    for nn in range(Mu_r.shape[0]):
        R_out[nn] = update_r_single(y_r[nn,:],Mu_r[nn,:],R_pre_r[nn], a0, h, use_hmc)
    
    return R_out

######################################################
##### clustering

#### regular clustering
def logsumexp(a,b):
    m = max(a,b)
    if m == -np.Inf:
        out = -np.Inf
    else:
        out = np.log(np.exp(a-m) + np.exp(b-m)) + m
    return out

def MFMcoeff(log_pk, MFMgamma, n, upto):
    lAbsGam = lambda x: np.log(np.abs(gamma(x)))
    tol = 1e-12
    log_v = np.zeros((upto,))

    for t in range(1, upto+1):
        if t > n:
            log_v[t-1] = -np.Inf
            continue
        a,c,k,p = 0,-np.Inf,1,0
        while (np.abs(a-c) > tol or p < 1- tol):
            if k >= t:
                a=c
                b=lAbsGam(k+1)-lAbsGam(k-t+1) - lAbsGam(k*MFMgamma+n)+lAbsGam(k*MFMgamma) + log_pk(k)
                c = logsumexp(a,b)
            
            p = p + np.exp(log_pk(k))
            k = k+1
        
        log_v[t-1] = c
        
    return log_v

def ordered_insert(index, actList, t):
    j = t-1
    while (j>=0) and (actList[j]>index):
        actList[j+1] = actList[j]
        j = j-1
    actList[j+1] = index
    return actList

def ordered_next(actList):
    j = 0
    while actList[j] == j:
        j = j+1
    return j

def ordered_remove(index, actList, t):
    for j in range(t):
        if actList[j] >= index:
            actList[j] = actList[j+1]
    
    return actList   

def nbLogMar(y_nb, r_nb, X_nb, offset_nb):
    
    # OK, I just do Laplace approximation here...
    # to debug...
    # y_nb = Y_tmp[ii,:].ravel()
    # r_nb = r_tmp[ii]
    # X_nb = muX_b[cc][1:,:].T
    # offset_nb = muX_b[cc][0,:] + delta_tmp[ii,:]

    nb2_res = sm.GLM(y_nb, X_nb,family=
                     sm.families.NegativeBinomial(alpha = 1/r_nb),
                     offset = offset_nb).fit()
    
    logMar = nb2_res.llf + np.sum(norm.logpdf(nb2_res.params)) + 0.5*np.log(np.linalg.det(nb2_res.cov_params()))
    return logMar

def randp(p,k):
    s = 0
    for j in range(k):
        s = s + p[j]
    
    u = np.random.uniform()*s
    j = 0
    C = p[0]
    while u > C:
        j = j + 1
        C = C + p[j]
    return j

def randlogp(log_p, k):
    log_s = -np.Inf
    for j in range(k):
        log_s = logsumexp(log_s, log_p[j])
    p = copy.deepcopy(log_p)
    for j in range(k):
        p[j] = np.exp(log_p[j] - log_s)
    
    j = randp(p,k)
    return j


def update_cluster(Z_a, numClus_a, t_a, actList_a,
                   c_next_a,DPMM, alpha_random,
                   a, log_v, logNb, muX_a, delta_tmp,
                   Y_tmp, r_tmp, prior, alphaDP,sigma_alpha,t_max, dynamics_a, sample_tag):
    
    
    # to debug...
    # Z_a = Z_fit[gg-1,:].copy()
    # numClus_a = numClus_fit[gg-1,:].copy()
    # t_a = t_fit[gg-1].copy()
    # actList_a = actList_fit[gg-1,:].copy()
    # c_next_a = c_next
    # muX_a = muX_fit[gg].copy()
    # delta_tmp = delt_fit[gg,:].copy().reshape((-1, 1))
    # Y_tmp = y
    # r_tmp = r_fit[gg,:].copy()
    # dynamics_a = dynamics_fit[gg].copy()
    
    
    N = Y_tmp.shape[0]
    T = Y_tmp.shape[1]
    p = muX_a[0].shape[0]-1
    lAbsGam = lambda x: np.log(np.abs(gamma(x)))

    Z_b = copy.deepcopy(Z_a)
    numClus_b = copy.deepcopy(numClus_a)
    t_b = copy.deepcopy(t_a)
    actList_b = copy.deepcopy(actList_a)
    muX_b = copy.deepcopy(muX_a)
    c_next_b = c_next_a
    dynamics_b = copy.deepcopy(dynamics_a)
    
    if DPMM and alpha_random:
        # MH move for DP concentration parameter (using p_alpha(a) = exp(-a) = Exp(a|1))
        aprop = alphaDP*np.exp(np.random.uniform()*sigma_alpha)
        top = t_a*np.log(aprop) - lAbsGam(aprop+N) + lAbsGam(aprop) - aprop + np.log(aprop)
        bot = t_a*np.log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP) - alphaDP + np.log(alphaDP)
        if np.random.uniform() < min(1, np.exp(top-bot)):
            alphaDP = aprop
        log_v = np.arange(1, t_max+2)*np.log(alphaDP) - lAbsGam(alphaDP+N) + lAbsGam(alphaDP)
    
    for ii in range(N):
        
        # (a) remove point ii from its cluster
        c = Z_b[ii].copy()
        numClus_b[c] = numClus_b[c] - 1
        if numClus_b[c] > 0:
            c_prop = c_next_b
            if sample_tag == 1:
                muX_b[c_prop] = sample_muX(p, T)
            else:
                 muX_b[c_prop] = sample_muX2(p, T)
            # muX_b[c_prop] = jitters_muX(muX_b[c])
            
            ## for debugging...
            # muX_b[c_prop] = muX_all[0]
            
            
            ## to be safe...
            # while len(muX_b) <= c_prop:
            #     muX_b.append([])
            # muX_b[c_prop] = sample_muX(p, T)
            
        else:
            c_prop = c
            actList_b = ordered_remove(c, actList_b, t_b)
            t_b = t_b - 1
        
        # (b) compute probabilities for resampling
        log_p = np.zeros((t_b+1,))
        for j in range(t_b):
            cc = actList_b[j]
            # calculate the marginalized likelihood...
            try:
                logMar = nbLogMar(Y_tmp[ii,:].ravel(), r_tmp[ii], muX_b[cc][1:,:].T,
                              muX_b[cc][0,:] + delta_tmp[ii,:])
                
            except:
                logMar = -np.Inf
            
            log_p[j] = logNb[numClus_b[cc]-1] + logMar
        
        
        try:
            logMar = nbLogMar(Y_tmp[ii,:].ravel(), r_tmp[ii], muX_b[c_prop][1:,:].T,
                              muX_b[c_prop][0,:] + delta_tmp[ii,:])
                
        except:
            logMar = -np.Inf
        log_p[t_b] = log_v[t_b] - log_v[t_b-1] + np.log(a) + logMar
        
        # (c) sample a new cluster for it
        j = randlogp(log_p, t_b+1)
    
        # (d) add point i to its new clusters
        if (j+1) <= t_b:
            c = actList_b[j]
        else:
            c = c_prop
            idx_tmp = np.arange(c*(p+1), (c+1)*(p+1))
            for ss in range(dynamics_b['As'].shape[0]):
                dynamics_b['bs'][ss,idx_tmp] = np.zeros((p+1,))
                dynamics_b['As'][ss,idx_tmp,:] = 0
                dynamics_b['As'][ss,:,idx_tmp] = 0
                dynamics_b['As'][ss,idx_tmp,idx_tmp] = 1
                dynamics_b['Qs'][ss,idx_tmp,:] = 0
                dynamics_b['Qs'][ss,:,idx_tmp] = 0
                dynamics_b['Qs'][ss,idx_tmp,idx_tmp] = 1e-2
            
            actList_b = ordered_insert(c, actList_b, t_b)
            t_b = t_b + 1
            c_next_b = ordered_next(actList_b)
        
        Z_b[ii] = c
        numClus_b[c] = numClus_b[c] + 1
    return Z_b, numClus_b, t_b, actList_b, c_next_b, muX_b, dynamics_b 

#### split-and-merge

def update_factor_sm(muX_sm_a, muX_sm_b, delta_sm, C_sm, Y_tmp,prior,active,density, r_sm, dynamics_sm):
    
    # to debug
    # muX_sm_a = tm.copy()
    # muX_sm_b = tm.copy()
    # delta_sm = delt_m[mIdx]
    # C_sm = C_m[mIdx,:]
    # Y_tmp = Y_tmp[mIdx,:]
    # active = True
    # density = True
    # r_sm = r_tmp[mIdx]
    # dynamics_sm
    
    N_tmp = Y_tmp.shape[0]
    T = Y_tmp.shape[1]
    p = C_sm.shape[1]
    
    r_sm = r_sm.reshape((N_tmp,1))
    delta_sm = delta_sm.reshape((N_tmp,1))
    
    log_pdf = 0
    C_expand = np.column_stack((np.ones((C_sm.shape[0],)), C_sm))
    
    
    # to make things simple, only consider single state linear dynamics
    if active:
        # 1) PG-augmentation
        omega = random_polyagamma(r_sm + Y_tmp, delta_sm + C_expand @ muX_sm_b - np.log(r_sm))
        kappa = (Y_tmp-r_sm)/2 + omega*(np.log(r_sm) - delta_sm)
        y_hat = (1/omega)*kappa
        
        # 2) FFBS
        
        muX_sm_b = FFBS(y_hat,C_expand,1/omega,dynamics_sm['states'], dynamics_sm['As'],
                     dynamics_sm['bs'],dynamics_sm['Qs'],prior['x0'],prior['Q0'])
        
    if density:
        mu_tmp = np.exp(delta_sm + C_expand @ muX_sm_b)
        p_tmp = r_sm/(r_sm+mu_tmp)
        log_pdf = np.sum(nbinom.logpmf(Y_tmp, r_sm, p_tmp))
        
    return muX_sm_b, log_pdf

def restricted_gibbs_sm(zsa,zsb,tia,tib,tja,tjb,cia,cib,
                        cja,cjb,iIdx,jIdx,ni,nj,ism,jsm,
                        S,ns,Y,b,prior,active,delt_s,C_s,r_tmp,dynamics):
    
    # # to debug...
    # zsa = zs.copy()
    # zsb = zs.copy()
    # tia = ti.copy()
    # tib = ti.copy()
    # tja = tj.copy()
    # tjb = tj.copy()
    # cia = ci
    # cib = ci
    # cja = cj
    # cjb = cj
    # Y = Y_tmp
    # active = True
    # r_tmp
    # dynamics
    
    T = Y.shape[1]
    
    log_p = 0
    for ks in range(ns):
        k = S[ks]
        if k != ism and k != jsm:
            if zsa[k] == cia: ni = ni-1
            else: nj = nj-1
            
            lami = np.exp(tia[0,:] + C_s[k,:] @ tia[1:,:] + delt_s[k])
            p_i = r_tmp[k]/ (r_tmp[k] + lami)
            Li = np.sum(nbinom.logpmf(Y[k,:], r_tmp[k], p_i))
            
            lamj = np.exp(tja[0,:] + C_s[k,:] @ tja[1:,:] + delt_s[k])
            p_j = r_tmp[k]/ (r_tmp[k] + lamj)
            Lj = np.sum(nbinom.logpmf(Y[k,:], r_tmp[k], p_j))
            
            Pi = np.exp(np.log(ni+b) + Li - logsumexp(np.log(ni+b) + Li, np.log(nj+b) + Lj))
            
            if active:
                if np.random.uniform() < Pi:
                    if zsa[k] == cja:
                        jIdx = np.setdiff1d(jIdx, k)
                        iIdx = np.append(iIdx, k)
                    
                    zsb[k] = cib
                else:
                    if zsa[k] == cia:
                        iIdx = np.setdiff1d(iIdx, k)
                        jIdx = np.append(jIdx, k)
                        
                    zsb[k] = cjb
            
            if zsb[k] == cib: ni = ni+1; log_p = log_p + np.log(Pi)
            else: nj = nj+1; log_p = log_p + np.log(1-Pi)

    tib, log_pi = update_factor_sm(tia, tib, delt_s[iIdx], C_s[iIdx,:], Y[iIdx,:],prior,active,True, r_tmp[iIdx], dynamics)
    tjb, log_pj = update_factor_sm(tja, tjb, delt_s[jIdx], C_s[jIdx,:], Y[jIdx,:],prior,active,True, r_tmp[jIdx], dynamics)
    
    log_p = log_p + log_pi + log_pj
    
    return zsb,tib,tjb,cib,cjb,log_p, ni, nj, iIdx, jIdx

def log_prior_sm(muX, prior, delt, C, dynamics):
    
    # to debug...

    # muX = ti.copy()
    # delt = delt_s[mIdx].copy()
    # C = C_s[mIdx,:].copy()
    # dynamics

    NN = C.shape[0]
    p = muX.shape[0]
    T = muX.shape[1]
    
    lpdf = 0
    
    # 1) muX
    lpdf = lpdf + multivariate_normal.logpdf(muX[:,0],prior['x0'].ravel(),prior['Q0'])
    
    for tt in range(1,T):
        state_tmp = dynamics['states'][tt]
        A_tmp = dynamics['As'][state_tmp]
        b_tmp = dynamics['bs'][state_tmp]
        Q_tmp = dynamics['Qs'][state_tmp]
        
        lpdf = lpdf + multivariate_normal.logpdf(muX[:,tt], b_tmp + A_tmp @ muX[:,tt-1],Q_tmp)
        
    
    # 2) deltC
    prior_mudc = np.append(prior['mud0'], prior['muC0'])
    prior_sigdc = block_diag(prior['s2d0'], prior['SigC0'])
    deltC = np.column_stack((delt, C))
    
    for nn in range(NN):
        lpdf = lpdf + multivariate_normal.logpdf(deltC[nn,:], prior_mudc, prior_sigdc)
    
    return lpdf

def splitMerge(Z_a, numClus_a, t_a, actList_a,
    muX_a,delt_a,C_a,Y_tmp,r_tmp,dynamics_sm,prior,
    a, b, log_v,splitL, nSplit, nMerge, dynamics_a):
    
    # to debug
    # Z_a = Z_fit[gg-1,:].copy()
    # numClus_a = numClus_fit[gg-1,:].copy()
    # t_a = t_fit[gg-1].copy()
    # actList_a = actList_fit[gg-1,:].copy()
    # muX_a = muX_fit[gg].copy()
    # delt_a = delt_fit[gg,:].copy().reshape((-1, 1))
    # C_a = C_fit[gg,:,:].copy()
    # Y_tmp = y
    # r_tmp = r_fit[gg,:].copy()
    # dynamics_sm
    # splitL = 1
    # nSplit = 5
    # nMerge = 5
    # dynamics_a = dynamics_fit[gg]
    
    
    Z_b = copy.deepcopy(Z_a)
    numClus_b = copy.deepcopy(numClus_a)
    t_b = copy.deepcopy(t_a)
    actList_b = copy.deepcopy(actList_a)
    muX_b = copy.deepcopy(muX_a)
    delt_b = copy.deepcopy(delt_a)
    C_b = copy.deepcopy(C_a)
    dynamics_b = copy.deepcopy(dynamics_a)
    
    N = Y_tmp.shape[0]
    T = Y_tmp.shape[1]
    p = C_a.shape[1]
    lAbsGam = lambda x: np.log(np.abs(gamma(x)))
    
    # (a) randomly choose a pair of indices
    try:
        if ~np.isnan(splitL):
            if splitL:
                maxIdx = np.argmax(numClus_b)
                rdIdx = np.random.choice(np.where(Z_b == maxIdx)[0], 2, replace = False)
            else:
                sVal2 = heapq.nsmallest(2, numClus_b[numClus_b != 0])

                if len(sVal2) == 1:
                    rdIdx = np.random.choice(N,2,replace = False)
                else:
                    set1 = np.where(numClus_b == sVal2[0])[0]
                    minIdx1 = np.random.choice(set1,1)
                    set2 = np.setdiff1d(np.where(numClus_b == sVal2[1])[0], minIdx1)
                    minIdx2 = np.random.choice(set2,1)

                    rdIdx = np.random.choice(np.where(Z_b == minIdx1)[0],1)
                    rdIdx = np.append(rdIdx, np.random.choice(np.where(Z_b == minIdx2)[0],1))

        else:
            rdIdx = np.random.choice(N,2,replace = False)
    except:
        rdIdx = np.random.choice(N,2,replace = False)
        
            
    ism = rdIdx[0]
    jsm = rdIdx[1]
    
    ci0 = Z_b[ism]
    cj0 = Z_b[jsm]
    ti0 = muX_b[ci0]
    tj0 = muX_b[cj0]
    
    # (b) set S[0],...,S[ns-1] to the indices of points in clusters ci0 and cj0
    S = np.zeros((N,), dtype = int)
    ns = 0
    for k in range(N):
        if Z_b[k] == ci0 or Z_b[k] == cj0:
            ns = ns + 1
            S[ns-1] = k
    
    # (c) find available cluster IDs for merge and split parameters
    k = 0
    while actList_b[k] == k: k = k+1
    cm = k
    while actList_b[k] == k+1: k = k+1
    ci = k+1
    while actList_b[k] == k+2: k = k+1
    cj = k+2
    
    # (d) randomly choose the merge launch state
    tm = sample_muX(p,T)
    mIdx = S[0:ns]
    delt_m = copy.deepcopy(delt_b)
    C_m = copy.deepcopy(C_b)
    
    for mm in range(nMerge):
        
        # update factor
        tm,_ = update_factor_sm(tm.copy(), tm.copy(), delt_m[mIdx],
                               C_m[mIdx,:], Y_tmp[mIdx,:],prior,True,False, r_tmp[mIdx], dynamics_sm)
         
        # update loading
        delt_m[mIdx], C_m[mIdx,:] = update_deltC(delt_m[mIdx].copy(), C_m[mIdx,:].copy(), Y_tmp[mIdx,:],
                                                  r_tmp[mIdx], np.zeros((mIdx.size, ), dtype = int), [tm], prior)
        
        
    # (e) randomly choose the split lauch state
    ti = sample_muX(p,T)
    tj = sample_muX(p,T)
    
    zs = np.zeros((N,), dtype = int)
    zs[ism] = ci
    zs[jsm] = cj
    
    kOut = np.setdiff1d(S[0:ns], np.array([ism,jsm]))
    splitIdx = np.random.binomial(1, 0.5, (kOut.size,))
    
    siIdx = kOut[splitIdx == 1]
    sjIdx = kOut[splitIdx == 0]
    zs[siIdx] = ci
    zs[sjIdx] = cj
    
    iIdx = np.append(ism, siIdx)
    jIdx = np.append(jsm, sjIdx)
    ni = iIdx.size
    nj = jIdx.size
    
    # make several moves (restricted Gibbs)
    delt_s = delt_b.copy()
    C_s = C_b.copy()
    
    for ss in range(nSplit):
        
        # update factor
        zs,ti,tj,ci,cj,_, ni, nj, iIdx, jIdx = restricted_gibbs_sm(zs.copy(),zs.copy(),
                                                                   ti.copy(),ti.copy(),tj.copy(),tj.copy(),
                                                                   ci,ci,cj,cj,iIdx,jIdx,ni,nj,ism,jsm,
                                                                   S,ns,Y_tmp,b,prior,True,delt_s,C_s,r_tmp,dynamics_sm)
        
        
        # update loading
        delt_s[iIdx], C_s[iIdx,:] = update_deltC(delt_s[iIdx].copy(), C_s[iIdx,:].copy(), Y_tmp[iIdx,:],
                                                  r_tmp[iIdx], np.zeros((iIdx.size, ), dtype = int), [ti], prior)
        
        delt_s[jIdx], C_s[jIdx,:] = update_deltC(delt_s[jIdx].copy(), C_s[jIdx,:].copy(), Y_tmp[jIdx,:],
                                                  r_tmp[jIdx], np.zeros((jIdx.size, ), dtype = int), [tj], prior)
        
        
    # (f) make proposal
    if ci0 == cj0: # propose a split
        # make one final sweep and compute its density
        
        # update factors
        zs,ti,tj,ci,cj,log_prop_ab, ni, nj, iIdx, jIdx = restricted_gibbs_sm(zs.copy(),zs.copy(),
                                                                             ti.copy(),ti.copy(),tj.copy(),tj.copy(),
                                                                             ci,ci,cj,cj,iIdx,jIdx,ni,nj,ism,jsm,
                                                                             S,ns,Y_tmp,b,prior,True,delt_s,C_s,r_tmp,dynamics_sm)
        
        
        # update loadings
        
        delt_s[iIdx], C_s[iIdx,:] = update_deltC(delt_s[iIdx].copy(), C_s[iIdx,:].copy(), Y_tmp[iIdx,:],
                                                  r_tmp[iIdx], np.zeros((iIdx.size, ), dtype = int), [ti], prior)
        
        delt_s[jIdx], C_s[jIdx,:] = update_deltC(delt_s[jIdx].copy(), C_s[jIdx,:].copy(), Y_tmp[jIdx,:],
                                                  r_tmp[jIdx], np.zeros((jIdx.size, ), dtype = int), [tj], prior)
        
        # compute density of Lmerge to original state
        ti0, log_prop_ba = update_factor_sm(tm.copy(), ti0.copy(), delt_a[mIdx].copy(),
                                           C_a[mIdx,:].copy(),Y_tmp[mIdx,:], prior, False, True,
                                           r_tmp[mIdx], dynamics_sm)
        
        
        # compute acceptance probability
        log_prior_b = log_v[t_b] + lAbsGam(ni+b) + lAbsGam(nj+b) - 2*lAbsGam(a) + \
        log_prior_sm(ti, prior, delt_s[iIdx,:], C_s[iIdx,:], dynamics_sm)
        
        
        log_prior_a = log_v[t_b-1] + lAbsGam(ns+b) - lAbsGam(a) + \
        log_prior_sm(ti0, prior, delt_s[Z_b == ci0,:], C_s[Z_b == ci0,:], dynamics_sm)
        
        llhd_ratio = 0
        for ks in range(ns):
            k = S[ks]
            if zs[k] == ci: lamTmp = np.exp(ti[0,:] + C_s[k,:] @ ti[1:,:] + delt_s[k])
            else: lamTmp = np.exp(tj[0,:] + C_s[k,:] @ tj[1:,:] + delt_s[k])
            
            pTmp = r_tmp[k]/ (r_tmp[k] + lamTmp)
            
            lamTmp0 = np.exp(ti0[0,:] + C_a[k,:] @ ti0[1:,:] + delt_a[k])
            pTmp0 = r_tmp[k]/ (r_tmp[k] + lamTmp0)
            
            llhd_ratio = llhd_ratio + np.sum(nbinom.logpmf(Y_tmp[k,:], r_tmp[k], pTmp)) - np.sum(nbinom.logpmf(Y_tmp[k,:], r_tmp[k], pTmp0))
        
        # p_accept = min(1, np.exp(log_prop_ba-log_prop_ab + log_prior_b-log_prior_a + llhd_ratio))
        # p_accept = min(1, np.exp(log_prior_b-log_prior_a + llhd_ratio))
        p_accept = min(1, np.exp(llhd_ratio))
        
        if np.random.uniform() < p_accept: # accept split
            print('accept split')
            for ks in range(ns):
                Z_b[S[ks]] = zs[S[ks]]
            actList_b = ordered_remove(ci0, actList_b, t_b)
            actList_b = ordered_insert(ci, actList_b, t_b-1)
            actList_b = ordered_insert(cj, actList_b, t_b)
            
            numClus_b[ci0] = 0
            numClus_b[ci] = ni
            numClus_b[cj] = nj
            t_b = t_b + 1
            
            idx_tmp = np.sort(np.append(np.arange(ci0*(p+1), (ci0+1)*(p+1)),
                                        np.append(np.arange(ci*(p+1), (ci+1)*(p+1)), 
                                                  np.arange(cj*(p+1), (cj+1)*(p+1)))))
            for ss in range(dynamics_b['As'].shape[0]):
                dynamics_b['bs'][ss,idx_tmp] = 0
                dynamics_b['As'][ss,idx_tmp,:] = 0
                dynamics_b['As'][ss,:,idx_tmp] = 0
                dynamics_b['As'][ss,idx_tmp,idx_tmp] = 1
                dynamics_b['Qs'][ss,idx_tmp,:] = 0
                dynamics_b['Qs'][ss,:,idx_tmp] = 0
                dynamics_b['Qs'][ss,idx_tmp,idx_tmp] = 1e-2
            
            
            muX_b[ci] = ti
            muX_b[cj] = tj
            
            delt_b = delt_s.copy()
            C_b = C_s.copy()
            
        else:
            print('reject split')
        
    else: # propose a merge
        # make one final sweep and compute its probability density
        
        # update factors
        tm,log_prop_ab = update_factor_sm(tm.copy(), tm.copy(), delt_m[mIdx],
                                          C_m[mIdx,:], Y_tmp[mIdx,:],prior,True,True, r_tmp[mIdx], dynamics_sm)
        
        
        # upadte loadings
        delt_m[mIdx], C_m[mIdx,:] = update_deltC(delt_m[mIdx].copy(), C_m[mIdx,:].copy(), Y_tmp[mIdx,:],
                                                  r_tmp[mIdx], np.zeros((mIdx.size, ), dtype = int), [tm], prior)
        
        # compute probability density of going from split launch state to original state
        _,_,_,_,_,log_prop_ba, ni, nj, iIdx, jIdx = restricted_gibbs_sm(zs.copy(),Z_b.copy(),
                                                                             ti.copy(),ti0.copy(),tj.copy(),tj0.copy(),
                                                                             ci,ci0,cj,cj0,iIdx,jIdx,ni,nj,ism,jsm,
                                                                             S,ns,Y_tmp,b,prior,False,delt_s,C_s,r_tmp,dynamics_sm)
        
        # compute acceptance probability
        log_prior_b = log_v[t_b-2] + lAbsGam(ns+b) - lAbsGam(a) + log_prior_sm(tm, prior, delt_s[mIdx,:], C_s[mIdx,:], dynamics_sm)
        log_prior_a = log_v[t_b-1] + lAbsGam(ni+b) + lAbsGam(nj+b) - 2*lAbsGam(a) + \
        log_prior_sm(ti0, prior, delt_s[Z_b == ci0,:], C_s[Z_b == ci0,:], dynamics_sm) +\
        log_prior_sm(tj0, prior, delt_s[Z_b == cj0,:], C_s[Z_b == cj0,:], dynamics_sm)
        
        llhd_ratio = 0
        for ks in range(ns):
            k = S[ks]
            if Z_b[k] == ci0: lamTmp0 = np.exp(ti0[0,:] + C_a[k,:] @ ti0[1:,:] + delt_a[k])
            else: lamTmp0 = np.exp(tj0[0,:] + C_a[k,:] @ tj0[1:,:] + delt_a[k])
                
            pTmp0 = r_tmp[k]/ (r_tmp[k] + lamTmp0)
            lamTmp = np.exp(tm[0,:] + C_m[k,:] @ tm[1:,:] + delt_m[k])
            pTmp = r_tmp[k]/ (r_tmp[k] + lamTmp)
            
            llhd_ratio = llhd_ratio + np.sum(nbinom.logpmf(Y_tmp[k,:], r_tmp[k], pTmp)) - np.sum(nbinom.logpmf(Y_tmp[k,:], r_tmp[k], pTmp0))
        
        
        p_accept = min(1, np.exp(log_prop_ba-log_prop_ab + log_prior_b-log_prior_a + llhd_ratio))
        # p_accept = min(1, np.exp(log_prior_b-log_prior_a + llhd_ratio))
        if np.random.uniform() < p_accept: # accept merge
            print('accept merge')
            for ks in range(ns):
                Z_b[S[ks]] = cm
                
            actList_b = ordered_remove(ci0, actList_b, t_b)
            actList_b = ordered_remove(cj0, actList_b, t_b-1)
            actList_b = ordered_insert(cm, actList_b, t_b-2)
            
            
            numClus_b[cm] = ns
            numClus_b[ci0] = 0
            numClus_b[cj0] = 0
            t_b = t_b - 1
            
            idx_tmp = np.sort(np.append(np.arange(cm*(p+1), (cm+1)*(p+1)),
                                        np.append(np.arange(ci0*(p+1), (ci0+1)*(p+1)), 
                                                  np.arange(cj0*(p+1), (cj0+1)*(p+1)))))
            for ss in range(dynamics_b['As'].shape[0]):
                dynamics_b['bs'][ss,idx_tmp] = 0
                dynamics_b['As'][ss,idx_tmp,:] = 0
                dynamics_b['As'][ss,:,idx_tmp] = 0
                dynamics_b['As'][ss,idx_tmp,idx_tmp] = 1
                dynamics_b['Qs'][ss,idx_tmp,:] = 0
                dynamics_b['Qs'][ss,:,idx_tmp] = 0
                dynamics_b['Qs'][ss,idx_tmp,idx_tmp] = 1e-2
            
            muX_b[cm] = tm
            delt_b = delt_m.copy()
            C_b = C_m.copy()
            
        else:
            print('reject merge')
    
    return Z_b, numClus_b, t_b, actList_b, muX_b, delt_b.ravel(), C_b, dynamics_b

######################################################
##### plotting
def clusterPlot(Y,z):
    
    # to debug...
    # Y = y
    # z = lab_neuron
    
    sorted_unique = np.sort(np.unique(z))
    k = sorted_unique.size
    
    reLabelTab = {sorted_unique[k]: k for k in np.arange(k)}
    z2 = np.zeros_like(z)
    for kk in range(len(z)):
        z2[kk] = reLabelTab[z[kk]]
    
    N = Y.shape[0]
    T = Y.shape[1]
    Yplot = Y + np.matlib.repmat(np.linspace(10*N, 10, N)[:,None],1,T)
    
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 0.9, k)]
    for mm in range(N):
        plt.plot(Yplot[mm,:], color = colors[z2[mm]])

def clusterPlot_values(Y,z):
    
    # to debug...
    # Y = y
    # z = lab_neuron
    
    sorted_unique = np.sort(np.unique(z))
    k = sorted_unique.size
    
    reLabelTab = {sorted_unique[k]: k for k in np.arange(k)}
    z2 = np.zeros_like(z)
    for kk in range(len(z)):
        z2[kk] = reLabelTab[z[kk]]
    
    N = Y.shape[0]
    T = Y.shape[1]
    Yplot = Y + np.matlib.repmat(np.linspace(10*N, 10, N)[:,None],1,T)
    colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 0.9, k)]
    
    return Yplot, z2, colors





