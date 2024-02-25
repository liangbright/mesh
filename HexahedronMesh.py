# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from torch_sparse import SparseTensor
import numpy as np
import json
from torch.linalg import det, cross
from PolyhedronMesh import PolyhedronMesh
#%%
class HexahedronMesh(PolyhedronMesh):
    #8-node C3D8 mesh
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polyhedron_hex8'

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
            edge.append([id0, id1])
            edge.append([id1, id2])
            edge.append([id2, id3])
            edge.append([id3, id0])
            edge.append([id4, id5])
            edge.append([id5, id6])
            edge.append([id6, id7])
            edge.append([id7, id4])
            edge.append([id0, id4])
            edge.append([id1, id5])
            edge.append([id2, id6])
            edge.append([id3, id7])            
        edge=np.array(edge, dtype=np.int64)
        edge=np.sort(edge, axis=1)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        self.element_to_edge_adj_table=inverse.reshape(-1,12).tolist()

    def build_face(self):
        #self.face[k] is a quad: [node_idx0, node_idx1, node_idx2, node_idx3]
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        face=[]
        for m in range(0, len(element)):
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=int(element[m][4])
            id5=int(element[m][5])
            id6=int(element[m][6])
            id7=int(element[m][7])
            face.append([id0, id3, id2, id1])
            face.append([id4, id5, id6, id7])
            face.append([id0, id1, id5, id4])
            face.append([id1, id2, id6, id5])
            face.append([id2, id3, id7, id6])
            face.append([id3, id0, id4, id7])
        face=np.array(face, dtype=np.int64)
        face=np.sort(face, axis=1)
        face_unique, inverse=np.unique(face, return_inverse=True, axis=0)
        self.face=torch.tensor(face_unique, dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,6).tolist()

    def upate_element_volume(self):
        self.element_volume=HexahedronMesh.cal_element_volume(self.node, self.element)
    
    @staticmethod
    def cal_element_volumn(node, element):
        #X=self.node[self.element]#shape (M,8,3)
        #r1i=get_integration_point_1i(X.dtype, X.device)
        #dX_dr=cal_dh_dr(r1i, X)
        #volumn=8*det(dX_dr)
        #---------no need for FEA element---
        #divide a hex8 to 12 tet4, and cal_vol of each tet4
        def cal_vol(node0, node1, node2, node3):
            a=node1-node0; b=node2-node0; c=node3-node0
            vol=(1/6)*(cross(a,b)*c).sum(dim=-1).abs() 
            return vol                   
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x4=node[element[:,4]]
        x5=node[element[:,5]]
        x6=node[element[:,6]]
        x7=node[element[:,7]]
        volume=(cal_vol(x1,x2,x3,x5)
               +cal_vol(x1,x5,x2,x3)
               +cal_vol(x4,x7,x5,x3)
               +cal_vol(x3,x7,x4,x5)
               +cal_vol(x0,x4,x5,x3)
               +cal_vol(x0,x3,x4,x5)
               +cal_vol(x2,x5,x6,x3)
               +cal_vol(x2,x6,x3,x5)
               +cal_vol(x5,x7,x6,x3)
               +cal_vol(x3,x6,x7,x5)
               +cal_vol(x0,x5,x1,x3)
               +cal_vol(x0,x1,x3,x5))
        return volume

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        #element.shape (M,8)
        element_sub=self.element[element_idx_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idx_list]
        element_out=element_out.view(len(element_idx_list),-1)
        mesh_new=HexahedronMesh(node_out, element_out)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    pass
