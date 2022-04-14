# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
import numpy as np
import json
from torch.linalg import det
from Element_C3D8 import cal_dh_dr, get_integration_point_1i, get_integration_point_8i, interpolate
from PolyhedronMesh import PolyhedronMesh
_Flag_VTK_IMPORT_=False
try:
    import vtk
    from vtk import vtkPoints
    from vtk import vtkPolyData, vtkPolyDataReader, vtkPolyDataWriter
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class HexMesh(PolyhedronMesh):
    #8-node C3D8 mesh
    def __init__(self):
        super().__init__()

    def cal_element_volumn(self):
        X=self.node[self.element]#shape (M,8,3)
        r1i=get_integration_point_1i(X.dtype, X.device)
        dX_dr=cal_dh_dr(r1i, X)
        volumn=8*det(dX_dr)
        return volumn

    @staticmethod
    def subdivide(node, element):
        pass
#%%
def get_sub_mesh(node, element, element_sub):
    #element.shape (M,8)
    node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
    node_out=node[node_idlist]
    element_out=element_out.view(-1,8)
    return node_out, element_out
#%%
if __name__ == "__main__":
    pass
