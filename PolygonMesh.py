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
    # use this class to handle mixture of triangle and quad elements
    def __init__(self):
        super().__init__('polygon')

    def build_adj_node_link(self, undirected):
        #undirected: True or False
        adj_node_link=[]
        if undirected == True:
            for m in range(0, len(self.element)):
                e=self.element[m]
                for k in range(0, len(e)):
                    if k < len(e)-1:
                        a=e[k]; b=e[k+1]
                    else:
                        a=e[k]; b=e[0]
                    adj_node_link.append([a, b])
                    adj_node_link.append([b, a])
        else:
            for m in range(0, len(self.element)):
                e=self.element[m]
                for k in range(0, len(e)):
                    if k < len(e)-1:
                        a=e[k]; b=e[k+1]
                    else:
                        a=e[k]; b=e[0]
                    if a < b:
                        adj_node_link.append([a, b])
                    else:
                        adj_node_link.append([b, a])
        adj_node_link=torch.tensor(adj_node_link, dtype=torch.int64)
        adj_node_link=torch.unique(adj_node_link, dim=0, sorted=True)
        if undirected == True:
            self.adj_node_link['undirected']=adj_node_link
        else:
            self.adj_node_link['directed']=adj_node_link

    def update_edge_length(self):
         self.edge_length=PolygonMesh.cal_edge_length(self.node, self.adj_node_link)

    @staticmethod
    def cal_edge_length(node, adj_node_link):
        x_j=node[adj_node_link[:,0]]
        x_i=node[adj_node_link[:,1]]
        edge_length=torch.norm(x_i-x_j, p=2, dim=1, keepdim=True)
        return edge_length
#%%
if __name__ == "__main__":
    filename="F:/MLFEA/TAA/data/343c1.5/bav17_AortaModel_P0_best.vtk"
    root=PolygonMesh()
    root.load_from_vtk(filename, torch.float32)
    root.node_data={'node_data1':torch.rand((len(root.node), 6)),
                     'node_data2':torch.rand((len(root.node), 6))}
    root.element_data={'element_data1':torch.rand((len(root.element), 6)),
                       'element_data2':torch.rand((len(root.element), 6))}
    root.save_by_vtk("F:/MLFEA/TAA/test_poly.vtk")
