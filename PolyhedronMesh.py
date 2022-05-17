# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from Mesh import Mesh
#%%
class PolyhedronMesh(Mesh):
    def __init__(self):
        super().__init__('polyhedron')

#%%
def get_sub_mesh_hex8(node, element, element_sub):
    #element.shape (M,8)
    node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
    node_out=node[node_idlist]
    element_out=element_out.view(-1,8)
    return node_out, element_out
#%%
if __name__ == "__main__":
    #
    filename="D:/MLFEA/TAVR/FE/1908788_0_im_5_phase1_Root_solid_three_layers_aligned.vtk"
    root=PolyhedronMesh()
    root.load_from_vtk(filename, dtype=torch.float32)
    #
    root.node_name_to_index["C01"]=131
    root.node_name_to_index["C12"]=2010
    root.node_name_to_index["C20"]=1
    root.node_name_to_index["H0"]=2
    root.node_name_to_index["H1"]=1883
    root.node_name_to_index["H2"]=3759
    root.element_set['Leaflet1']=np.arange(0, 888)
    root.element_set['Leaflet2']=np.arange(888, 2*888)
    root.element_set['Leaflet3']=np.arange(2*888, 3*888)
    #
    node_stress=np.random.rand(root.node.shape[0], 9)
    root.node_data["stress"]=node_stress
    element_stress=np.random.rand(len(root.element), 9)
    root.element_data["stress"]=element_stress
    #
    root.save_by_vtk("test.vtk")
    #
    root=PolyhedronMesh()
    root.load_from_vtk("test.vtk", dtype=torch.float32)
