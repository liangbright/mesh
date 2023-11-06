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
from torch.linalg import det
try:
    from Element_C3D8 import cal_dh_dr, get_integration_point_1i, get_integration_point_8i, interpolate
except:
    print('can not import Element_C3D8 @ HexahedronMesh')
from PolyhedronMesh import PolyhedronMesh
#%%
class HexahedronMesh(PolyhedronMesh):
    #8-node C3D8 mesh
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polyhedron_hex8'

    def build_edge(self):
        edge=[]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(element)):
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=int(element[m][4])
            id5=int(element[m][5])
            id6=int(element[m][6])
            id7=int(element[m][7])
            if id0 < id1:
                edge.append([id0, id1])
            else:
                edge.append([id1, id0])
            if id1 < id2:
                edge.append([id1, id2])
            else:
                edge.append([id2, id1])
            if id2 < id3:
                edge.append([id2, id3])
            else:
                edge.append([id3, id2])
            if id3 < id0:
                edge.append([id3, id0])
            else:
                edge.append([id0, id3])
            if id4 < id5:
                edge.append([id4, id5])
            else:
                edge.append([id5, id4])
            if id5 < id6:
                edge.append([id5, id6])
            else:
                edge.append([id6, id5])
            if id6 < id7:
                edge.append([id6, id7])
            else:
                edge.append([id7, id6])
            if id7 < id4:
                edge.append([id7, id4])
            else:
                edge.append([id4, id7])
            if id0 < id4:
                edge.append([id0, id4])
            else:
                edge.append([id4, id0])
            if id1 < id5:
                edge.append([id1, id5])
            else:
                edge.append([id5, id1])
            if id2 < id6:
                edge.append([id2, id6])
            else:
                edge.append([id6, id2])
            if id3 < id7:
                edge.append([id3, id7])
            else:
                edge.append([id7, id3])
        edge=torch.tensor(edge, dtype=torch.int64)
        edge=torch.unique(edge, dim=0, sorted=True)
        self.edge=edge
        self.build_map_node_pair_to_edge()

    def build_element_to_edge_adj_table(self):
        if self.map_node_pair_to_edge is None:
            self.build_map_node_pari_to_edge()
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        adj_table=[[] for _ in range(len(self.element))]
        for m in range(0, len(element)):
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=int(element[m][4])
            id5=int(element[m][5])
            id6=int(element[m][6])
            id7=int(element[m][7])
            adj_table[m].append(self.get_edge_idx_from_node_pair(id0, id1))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id1, id2))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id2, id3))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id0, id3))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id4, id5))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id5, id6))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id6, id7))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id4, id7))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id0, id4))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id1, id5))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id2, id6))
            adj_table[m].append(self.get_edge_idx_from_node_pair(id3, id7))
        self.element_to_edge_adj_table=adj_table

    def build_face(self):
        #self.face[k] is a quad: [node_idx0, node_idx1, node_idx2, node_idx3]
        self.build_element_to_face_adj_table()

    def build_element_to_face_adj_table(self):
        face=[]
        face_sorted=[]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(self.element)):
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
        face_sorted,_=np.sort(face, axis=1)
        face_sorted_unique, index, inverse=np.unique(face_sorted, return_index=True, return_inverse=True, axis=0)
        self.face=torch.tensor(face[index], dtype=torch.int64)
        self.element_to_face_adj_table=torch.tensor(inverse.reshape(-1,6), dtype=torch.int64)

    def build_face_to_element_adj_table(self):
        if self.element_to_face_adj_table is None:
            self.build_element_to_face_adj_table()
        adj_table=[[] for _ in range(len(self.face))]
        for m in range(0, len(self.element)):
            face_idx_list=self.element_to_face_adj_table[m]
            for idx in face_idx_list:
                adj_table[idx].append(m)
        self.face_to_element_adj_table=adj_table

    def find_surface_face(self):
        if self.face_to_element_adj_table is None:
            self.build_face_to_element_adj_table()
        face_idx_list=[]
        for k in range(0, len(self.face)):
            adj_elm_idx=self.face_to_element_adj_table[k]
            if len(adj_elm_idx) <= 1:
                face_idx_list.append(k)
        return face_idx_list

    def find_surface_node(self):
        face_idx_list=self.find_surface_face()
        node_idx_list=[]
        for idx in face_idx_list:
            node_idx_list.extend(self.face[idx])
        node_idx_list=np.unique(node_idx_list).tolist()
        return node_idx_list

    def cal_element_volumn(self):
        X=self.node[self.element]#shape (M,8,3)
        r1i=get_integration_point_1i(X.dtype, X.device)
        dX_dr=cal_dh_dr(r1i, X)
        volumn=8*det(dX_dr)
        return volumn

    def subdivide(self):
        #draw a 3D figure and code this...
        pass

    def get_sub_mesh(self, element_idx_list):
        #element.shape (M,8)
        element_sub=self.element[element_idx_list]
        node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_out=self.node[node_idlist]
        element_out=element_out.view(-1,8)
        mesh_new=HexahedronMesh(node_out, element_out)
        return mesh_new
#%%
if __name__ == "__main__":
    pass
