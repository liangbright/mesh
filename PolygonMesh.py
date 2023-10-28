# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from copy import deepcopy
from Mesh import Mesh
import json
#%%
class PolygonMesh(Mesh):
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype, element_type=None, mesh_type='polygon')

    def save_as_mdk_json(self, filename):
        #to be compatible with MDK SavePolygonMeshFromJsonDataFile
        #use json to store node_set, element_set, etc
        #use vtk to store node, element, node_data, element_data, etc
        data={}
        data["Name"]=self.name
        data["ObjectType"]="PolygonMesh"
        data["ScalarType"]="double"
        data["IndexType"]="int_max"
        data["PointCount"]=self.node.shape[0]
        data["FaceCount"]=len(self.element)
        data["PointSetCount"]=len(self.node_set)
        data["FaceSetCount"]=len(self.element_set)
        data["PointSetList"]=[]
        for i, (k, v) in enumerate(self.node_set.items()):
            data["PointSetList"].append({k:v})
        data["FaceSetList"]=[]
        for i, (k, v) in enumerate(self.element_set.items()):
            data["FaceSetList"].append({k:v})
        data["PointAndFace"]=filename.split("/")[-1]+".vtk"
        with open(filename, "w") as outfile:
            json.dump(data, outfile, indent=4)
        self.save_as_vtk(filename+".vtk", ascii=True, vtk42=True, use_vtk=False)

    def build_edge(self):
        edge=[]
        for m in range(0, len(self.element)):
            elm=self.element[m]
            for k in range(0, len(elm)):
                if k < len(elm)-1:
                    a=int(elm[k]); b=int(elm[k+1])
                else:
                    a=int(elm[k]); b=int(elm[0])
                if a < b:
                    edge.append([a, b])
                else:
                    edge.append([b, a])
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
            elm=element[m]
            for n in range(0, len(elm)):
                idx_n=int(elm[n])
                if n < len(elm)-1:
                    idx_n1=int(elm[n+1])
                else:
                    idx_n1=int(elm[0])
                edge_idx=self.get_edge_idx_from_node_pair(idx_n, idx_n1)
                if edge_idx is None:
                    raise ValueError('edge_idx is None')
                adj_table[m].append(edge_idx)
        self.element_to_edge_adj_table=adj_table

    def update_edge_length(self):
        if self.edge is None:
            self.build_edge()
        self.edge_length=PolygonMesh.cal_edge_length(self.node, self.edge)

    @staticmethod
    def cal_edge_length(node, edge):
        x_j=node[edge[:,0]]
        x_i=node[edge[:,1]]
        edge_length=torch.norm(x_i-x_j, p=2, dim=1, keepdim=True)
        return edge_length

    def find_boundary_node(self):
        #return index list of nodes on boundary
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge_to_element_adj_table=self.edge_to_element_adj_table
        boundary=[]
        for k in range(0, len(self.edge)):
            elm=edge_to_element_adj_table[k]
            if len(elm) <= 1:
                boundary.append(int(self.edge[k,0]))
                boundary.append(int(self.edge[k,1]))
        boundary=np.unique(boundary).tolist()
        return boundary

    def is_quad(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 4:
                #this is QuadMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if m_min == m_max == 4:
            return True
        else:
            return False

    def is_tri(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3:
                #this is TriangleMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if m_min == m_max == 3:
            return True
        else:
            return False

    def is_quad_tri(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3 or len(self.element[0]) == 4:
                #this is QuadTriangleMesh
                return True
            else:
                return False
        m_list=[]
        for m in range(0, len(self.element)):
            m_list.append(len(self.element[m]))
        m_min=min(m_list)
        m_max=max(m_list)
        if (m_min == m_max == 3) or (m_min == m_max == 4):
            return True
        else:
            return False

    def quad_to_tri(self):
        #inplace function to divide every quad element to two triangle elements
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3:
                #this is TriangleMesh
                return
        element_new=[]
        m_list=[]
        flag=False
        for m in range(0, len(self.element)):
            elm=self.copy_to_list(self.element[m])
            if len(elm) == 4:
                #-----------
                # x3------x2
                # |       |
                # |       |
                # x0------x1
                # cut along x0-x2
                #-----------
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                id3=int(elm[3])
                element_new.append([id0, id2, id3])
                element_new.append([id0, id1, id2])
                m_list.append(3)
                flag=True
            elif len(elm) == 3:
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                element_new.append([id0, id1, id2])
                m_list.append(3)
            else:
                element_new.append(elm)
                m_list.append(len(elm))
        if flag == True: # at lest one quad is divided
            if min(m_list) == max(m_list):
                if isinstance(self.element, torch.Tensor):
                    element_new=torch.tensor(element_new, dtype=torch.int64, device=self.element.device)
                else:
                    element_new=torch.tensor(element_new, dtype=torch.int64)
            self.element=element_new
            self.clear_adj_info()

    def get_sub_mesh(self, element_idx_list):
        new_mesh=super().get_sub_mesh(element_idx_list)
        new_mesh=PolygonMesh(new_mesh.node, new_mesh.element)
        return new_mesh
#%%
if __name__ == "__main__":
    filename="D:/MLFEA/TAA/data/bav17_AortaModel_P0_best.pt"
    aorta=PolygonMesh()
    #aorta.load_from_vtk(filename, "float32")
    aorta.load_from_torch(filename)
    aorta.node_data={'node_data1':torch.rand((len(aorta.node), 6)),
                     'node_data2':torch.rand((len(aorta.node), 6))}
    aorta.element_data={'element_data1':torch.rand((len(aorta.element), 6)),
                       'element_data2':torch.rand((len(aorta.element), 6))}
    aorta.save_by_vtk("F:/MLFEA/TAA/test_poly.vtk")
    #%%
    from time import time
    t0=time()
    aorta.build_node_adj_link()
    #aorta.build_edge()
    t1=time()
    aorta.build_node_to_edge_table()
    t2=time()
    print(t1-t0, t2-t1)
    #%%
    t0=time()
    boundary=aorta.find_boundary_node()
    t1=time()
    print(t1-t0)
    #%%
    t0=time()
    aorta.build_node_to_node_table()
    t1=time()
    print(t1-t0)
    #%%
    t0=time()
    aorta.build_element_to_element_table(adj=2)
    t1=time()
    print(t1-t0)

