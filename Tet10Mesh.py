# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 18:30:52 2024

@author: liang
"""

import torch
#import torch_scatter
import numpy as np
from torch.linalg import det, cross
from PolyhedronMesh import PolyhedronMesh
#%%
class Tet10Mesh(PolyhedronMesh):
    #10-node C3D10/TET10 mesh
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element)
        self.mesh_type='polyhedron_tet10'

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=element[m]
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
    
    def build_face(self):
        #self.face[k] is a "triangle": [node_idx0, node_idx01, node_idx1, node_idx12, node_idx2, node_idx20]
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        face=[]
        for m in range(0, len(element)):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=element[m]
            face.append([id0, id6, id2, id5, id1, id4])
            face.append([id0, id4, id1, id8, id3, id7])
            face.append([id0, id7, id3, id9, id2, id6])
            face.append([id1, id5, id2, id9, id3, id8])
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=inverse.reshape(-1,4).tolist()
    
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
