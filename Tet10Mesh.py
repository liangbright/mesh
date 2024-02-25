# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:30:52 2024

@author: liang
"""

import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
from torch.linalg import det, cross
from PolyhedronMesh import PolyhedronMesh
#%%
class Tet10Mesh(PolyhedronMesh):
    #10-node C3D10/TET10 mesh
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polyhedron_tet10'

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        edge=[]
        for m in range(0, len(element)):
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=int(element[m][4])
            id5=int(element[m][5])
            id6=int(element[m][6])
            id7=int(element[m][7])
            id8=int(element[m][8])
            id9=int(element[m][9])
            edge.append([id0, id4])
            edge.append([id0, id6])
            edge.append([id0, id7])
            edge.append([id1, id4])
            edge.append([id1, id5])
            edge.append([id1, id8])
            edge.append([id2, id5])
            edge.append([id2, id6])
            edge.append([id2, id9])
            edge.append([id3, id7])
            edge.append([id3, id8])
            edge.append([id3, id9])
        edge=np.array(edge, dtype=np.int64)
        edge=np.sort(edge, axis=1)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        self.element_to_edge_adj_table=inverse.reshape(-1,12).tolist()
    
    def upate_element_volume(self):
        self.element_volume=Tet10Mesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volume(node, element):
        #tet4 formula
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        a=x1-x0
        b=x2-x0
        c=x3-x0
        volume=(1/6)*(cross(a,b)*c).sum(dim=-1).abs() #shape (M,)
        return volume

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        #element.shape (M,10)
        element_sub=self.element[element_idx_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idx_list]
        element_out=element_out.view(len(element_idx_list),-1)
        mesh_new=Tet10Mesh(node_out, element_out)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    pass
