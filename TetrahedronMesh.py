# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
from torch.linalg import det
from PolyhedronMesh import PolyhedronMesh
#%%
class TetrahedronMesh(PolyhedronMesh):
    #4-node C3D4/TET4 mesh
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polyhedron_tet4'

    def build_edge(self):
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        edge=[]
        for m in range(0, len(element)):
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=element[m][3]
            if id0 < id1:
                edge.append([id0, id1])
            else:
                edge.append([id1, id0])
            if id0 < id2:
                edge.append([id0, id2])
            else:
                edge.append([id2, id0])
            if id0 < id3:
                edge.append([id0, id3])
            else:
                edge.append([id3, id0])
            if id1 < id2:
                edge.append([id1, id2])
            else:
                edge.append([id2, id1])
            if id1 < id3:
                edge.append([id1, id3])
            else:
                edge.append([id3, id1])
            if id2 < id3:
                edge.append([id2, id3])
            else:
                edge.append([id3, id2])
        edge=torch.tensor(edge, dtype=torch.int64)
        edge=torch.unique(edge, dim=0, sorted=True)
        self.edge=edge

    def cal_element_volumn(self):
        #need C3D4 element
        pass

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        #element.shape (M,4)
        element_sub=self.element[element_idx_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idx_list]
        element_out=element_out.view(-1,4)
        mesh_new=TetrahedronMesh(node_out, element_out)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    pass
