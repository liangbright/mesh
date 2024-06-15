# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:40:25 2024

@author: liang
"""
import torch
import numpy as np
from copy import deepcopy
from Mesh import Mesh
#%%
class PolylineMesh(Mesh):
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element, element_type=None, mesh_type='polyline')        

    def initialize(self, curve_list):
        #curve_list is a list of curves, each cure is a numpy/torch array
        if len(curve_list)==0:
            return
        node=[]
        element=[]
        dtype=None
        for k in range(0, len(curve_list)):
            curve=curve_list[k]        
            if isinstance(curve, torch.Tensor) or isinstance(curve, np.ndarray):
                dtype=curve.dtype
                curve=curve.tolist()
            else:
                raise ValueError
            if len(curve) > 0:
                idx_start=len(node)
                node.extend(curve)
                element.append(np.arange(idx_start, idx_start+len(curve)).tolist())        
        self.node=torch.tensor(node, dtype=dtype)
        self.element=element
    