# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:40:25 2024

@author: liang
"""
import torch
import numpy as np
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
            if torch.is_tensor(curve):
                dtype=curve.dtype
                curve=curve.tolist()
            elif isinstance(curve, np.ndarray):
                if curve.dtype == np.float64:
                    dtype=torch.float64
                else:
                    dtype=torch.float32
                curve=curve.tolist()
            elif isinstance(curve, list) or isinstance(curve, tuple):
                dtype=torch.float32
            else:
                raise ValueError("unsupported type of curve_list")
            if len(curve) > 0:
                idx_start=len(node)
                node.extend(curve)
                idx_list=np.arange(idx_start, idx_start+len(curve)).tolist()
                for n in range(1, len(idx_list)):
                    element.append([idx_list[n-1], idx_list[n]]) #each element is VTK_LINE
        self.node=torch.tensor(node, dtype=dtype)
        self.element=element
    
    def build_edge(self):
        self.build_element_to_edge_adj_table()
        
    def build_element_to_edge_adj_table(self):
        #an edge connects 2 nodes
        #an element may be a simple VTK_LINE of 2 nodes or VTK_POLY_LINE of many nodes
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
        #a boundary node has no more than one neighbor node
        if self.node_to_node_adj_table is None:
            self.build_node_to_node_adj_table()
        boundary=[]
        for k in range(0, len(self.node)):
            adj_node_idx=self.node_to_node_adj_table[k]
            if len(adj_node_idx) <= 1:
                boundary.append(k)
        return boundary

    def find_boundary_edge(self):
        if self.edge is None:
            self.build_edge()
        boundary_node=self.find_boundary_node()
        boundary_edge=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if (idx0 in boundary_node) or (idx1 in boundary_node):
                boundary_edge.append(k)
        return boundary_edge

    def find_boundary_node_and_edge(self):
        if self.edge is None:
            self.build_edge()
        boundary_node=self.find_boundary_node()
        boundary_edge=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if (idx0 in boundary_node) or (idx1 in boundary_node):
                boundary_edge.append(k)
        return boundary_node, boundary_edge

    def find_boundary_element(self):
        #return index list of elements on boundary
        boundary_node=self.find_boundary_node()
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        boundary=[]
        for node_idx in boundary_node:
            adj_elm_idx=self.node_to_element_adj_table[node_idx]
            boundary.extend(adj_elm_idx)
        boundary=np.unique(boundary).tolist()
        return boundary

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        new_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        new_mesh=PolylineMesh(new_mesh.node, new_mesh.element)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
