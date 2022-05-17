# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from Mesh import Mesh
#%%
class PolygonMesh(Mesh):
    def __init__(self):
        super().__init__('polygon')

    def update_edge_length(self):
         self.edge_length=PolygonMesh.cal_edge_length(self.node, self.adj_node_link)

    @staticmethod
    def cal_edge_length(node, adj_node_link):
        x_i=node[adj_node_link[:,0]]
        x_j=node[adj_node_link[:,1]]
        edge_length=torch.norm(x_j-x_i, p=2, dim=1, keepdim=True)
        return edge_length
#%%
if __name__ == "__main__":
    filename="F:/MLFEA/TAA/data/343c1.5/bav17_AortaModel_P0_best.vtk"
    root=PolygonMesh()
    root.load_from_vtk(filename, torch.float32)
    root.save_by_vtk("F:/MLFEA/TAA/p0_ssm/test_poly.vtk")
