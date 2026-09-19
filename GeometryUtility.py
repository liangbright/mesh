#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 21:57:45 2026

@author: liang
"""
import torch
from torch.linalg import vector_norm as norm
import numpy as np
#%%
def ComputeAngleBetweenTwoVectorIn3D_slow(VectorA, VectorB):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.view(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.view(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        eps1=1e-12
        eps2=1e-7 # torch.acos grad issue, it must be 1e-7        
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        eps1=1e-12
        eps2=1e-16
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = np.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if np.any(L2Norm_A <= eps1) or np.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, np.clip to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        L2Norm_A=np.clip(L2Norm_A, min=eps1)
        L2Norm_B=np.clip(L2Norm_B, min=eps1)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, min=-1, max=1)
        Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA,  torch.Tensor):
        L2Norm_A = torch.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = torch.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if torch.any(L2Norm_A <= eps1) or torch.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, torch.clamp to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        with torch.no_grad():
            L2Norm_A.data.clamp_(min=eps1)
            L2Norm_B.data.clamp_(min=eps1)                
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);              
        CosTheta = torch.clamp(CosTheta, min=-1+eps2, max=1-eps2)
        Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
    return Theta
    '''
    #https://github.com/pytorch/pytorch/issues/8069  
    eps=1e-16
    x=torch.tensor(-1.0, requires_grad=True, dtype=torch.float64)
    x1=torch.clamp(x, min=-1+eps, max=1-eps)
    Theta = torch.acos(x1)
    Theta.backward()
    print(x.grad)
    '''
#%%
def cal_angle_between_3d_vector(VectorA, VectorB, return_cos=False):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.reshape(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.reshape(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        eps1=1e-12
        eps2=1e-7 # torch.acos grad issue, it must be 1e-7        
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        eps1=1e-12
        eps2=1e-16
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.linalg.norm(VectorA, ord=2, axis=-1)
        L2Norm_B = np.linalg.norm(VectorB, ord=2, axis=-1)
        if np.any(L2Norm_A <= eps1) or np.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, np.clip to eps @ cal_angle_between_3d_vector(...)")
        L2Norm_A=np.clip(L2Norm_A, a_min=eps1, a_max=np.inf)
        L2Norm_B=np.clip(L2Norm_B, a_min=eps1, a_max=np.inf)
        CosTheta = np.sum(VectorA*VectorB, axis=-1)/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, a_min=-1, a_max=1)
        if return_cos == False:
            Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA, torch.Tensor):
        L2Norm_A=norm(VectorA, ord=2, dim=-1)
        L2Norm_B=norm(VectorB, ord=2, dim=-1)        
        if torch.any(L2Norm_A <= eps1) or torch.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, torch.clamp to eps @ cal_angle_between_3d_vector(...)")
        with torch.no_grad():
            L2Norm_A.data.clamp_(min=eps1)
            L2Norm_B.data.clamp_(min=eps1)         
        CosTheta = (VectorA*VectorB).sum(dim=-1)/(L2Norm_A*L2Norm_B);                      
        if return_cos == False:
            CosTheta = torch.clamp(CosTheta, min=-1+eps2, max=1-eps2)
            Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
    else:
        raise ValueError('invalid input')
#%%
ComputeAngleBetweenTwoVectorIn3D=cal_angle_between_3d_vector