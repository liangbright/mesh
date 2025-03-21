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
    def __init__(self, node=None, element=None):
        super().__init__(node=node, element=element, element_type=None, mesh_type='polygon')
        
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
        self.save_as_vtk(filename+".vtk", vtk42=True, use_vtk=False)

    def load_from_mdk_json(self, filename, dtype):
        self.load_from_vtk(filename+".vtk", dtype)
        with open(filename) as f:
            data=json.load(f)
        for k in range(0, data["PointSetCount"]):
            temp=data["PointSetList"][k]
            name=list(temp.keys())[0]
            value=list(temp.values())[0]
            self.node_set[name]=value
        for k in range(0, data["FaceSetCount"]):
            temp=data["FaceSetList"][k]
            name=list(temp.keys())[0]
            value=list(temp.values())[0]
            self.element_set[name]=value
        self.name=data["Name"]

    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        element=self.element
        if not isinstance(element, list):
            element=element.tolist()
        edge=[]
        for m in range(0, len(element)):
            elm=element[m]
            for k in range(0, len(elm)):
                if k < len(elm)-1:
                    a=elm[k]; b=elm[k+1]
                else:
                    a=elm[k]; b=elm[0]
                if a < b:
                    edge.append([a, b])
                else:
                    edge.append([b, a])
        edge=np.array(edge, dtype=np.int64)
        edge_unique, inverse=np.unique(edge, return_inverse=True, axis=0)
        self.edge=torch.tensor(edge_unique, dtype=torch.int64)
        adj_table=[]
        idx=0
        for m in range(0, len(element)):
            adj_table.append(inverse[idx:(idx+len(element[m]))].tolist())
            idx=idx+len(element[m])        
        self.element_to_edge_adj_table=adj_table

    def find_boundary_node(self):
        #return index list of nodes on boundary
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary.append(edge[k,0])
                boundary.append(edge[k,1])
        boundary=np.unique(boundary).tolist()
        return boundary
    
    def find_boundary_edge(self):
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary.append(k)
        return boundary
    
    def find_boundary_node_and_edge(self):
        if self.edge is None:
            self.build_edge()
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        edge=self.edge.detach().cpu().numpy()
        boundary_node=[]
        boundary_edge=[]
        for k in range(0, len(edge)):
            adj_elm_idx=self.edge_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                boundary_node.append(edge[k,0])
                boundary_node.append(edge[k,1])
                boundary_edge.append(k)
        boundary_node=np.unique(boundary_node).tolist()
        return boundary_node, boundary_edge
    
    def find_boundary_element(self, adj):
        #return index list of elements on boundary
        if adj == "edge":
            if self.edge is None:
                self.build_edge()
            if self.edge_to_element_adj_table is None:
                self.build_edge_to_element_adj_table()
            edge=self.edge.detach().cpu().numpy()
            boundary=[]
            for k in range(0, len(edge)):
                adj_elm_idx=self.edge_to_element_adj_table[k]
                if len(adj_elm_idx) == 1:
                    boundary.append(adj_elm_idx[0])
            boundary=np.unique(boundary).tolist()
            return boundary
        elif adj == "node":
            boundary_node=self.find_boundary_node()
            if self.node_to_element_adj_table is None:
                self.build_node_to_element_adj_table()
            boundary=[]
            for node_idx in boundary_node:
                adj_elm_idx=self.node_to_element_adj_table[node_idx]
                boundary.extend(adj_elm_idx)
            boundary=np.unique(boundary).tolist()
            return boundary
        else:
            raise ValueError
        
    def is_quad(self):
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 4:
                #this is QuadMesh
                return True
            else:
                return False
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
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
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
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
        m_min=len(self.element[0])
        m_max=len(self.element[0])
        for m in range(1, len(self.element)):
            m_min=min(m_min, len(self.element[m]))
            m_max=max(m_max, len(self.element[m]))
        if (m_min == m_max == 3) or (m_min == m_max == 4):
            return True
        else:
            return False

    def quad_to_tri(self, mode=0):
        #divide every quad element to two triangle elements        
        if isinstance(self.element, torch.Tensor):
            if len(self.element[0]) == 3:                
                new_mesh=PolygonMesh(self.node.clone(), self.element.clone())
                return new_mesh
        element_new=[]
        m_list=[]
        for m in range(0, len(self.element)):
            elm=self.copy_to_list(self.element[m])
            if len(elm) == 4:
                #-----------
                # x3------x2
                # |       |
                # |       |
                # x0------x1
                # mode=0: cut along x0-x2
                # mode=1: cut along x1-x3
                #-----------
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                id3=int(elm[3])
                if mode == 0:
                    element_new.append([id0, id2, id3])
                    element_new.append([id0, id1, id2])
                elif mode == 1:
                    element_new.append([id0, id1, id3])
                    element_new.append([id1, id2, id3])
                elif mode == 'auto':
                    L02=((self.node[id0]-self.node[id2])**2).sum()
                    L13=((self.node[id1]-self.node[id3])**2).sum()
                    if L02 < L13:
                        element_new.append([id0, id2, id3])
                        element_new.append([id0, id1, id2])
                    else:
                        element_new.append([id0, id1, id3])
                        element_new.append([id1, id2, id3])
                else:
                    raise ValueError("mode is invalid")
                m_list.append(3)
            elif len(elm) == 3:
                id0=int(elm[0])
                id1=int(elm[1])
                id2=int(elm[2])
                element_new.append([id0, id1, id2])
                m_list.append(3)
            else:
                element_new.append(elm)
                m_list.append(len(elm))   
        new_mesh=PolygonMesh(self.node.clone(), element_new)   
        return new_mesh

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        new_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        new_mesh=PolygonMesh(new_mesh.node, new_mesh.element)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
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

