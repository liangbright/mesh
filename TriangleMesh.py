# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
import numpy as np
import json
from PolygonMesh import PolygonMesh
#%%
class TriangleMesh(PolygonMesh):

    def __init__(self):
        super().__init__()

    def update_node_normal(self):
        self.node_normal=TriangleMesh.cal_node_normal(self.node, self.element)

    @staticmethod
    def cal_node_normal(node, element, element_normal=None):
        if element_normal is None:
            element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element)
        M=element.shape[0]
        e_normal=element_normal.view(M, 1, 3)
        e_normal=e_normal.expand(M, 3, 3)
        e_normal=e_normal.reshape(M*3, 3)
        N=node.shape[0]
        normal = torch_scatter.scatter(e_normal, element.view(-1), dim=0, dim_size=N, reduce="sum")
        normal=normal/torch.norm(normal, p=2, dim=1, keepdim=True)
        normal=normal.contiguous()
        return normal

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element)

    @staticmethod
    def cal_element_area_and_normal(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #   x2
        #  /  \
        # x0--x1
        temp1=torch.cross(x1 - x0, x2- x0)
        temp2=torch.norm(temp1, p=2, dim=1, keepdim=True)
        area=0.5*temp2.abs()
        temp2=temp2.clamp(min=1e-12)
        normal=temp1/temp2
        return area, normal

    def sample_points_on_elements(self, n_points):
         return TriangleMesh.sample_points(self.node, self.element, n_points)

    @staticmethod
    def sample_points(node, element, n_points):
        area=TriangleMesh.cal_element_area(node, element)
        prob = area / area.sum()
        sample = torch.multinomial(prob, n_points-len(element), replacement=True)
        #print("sample_points", area.shape, prob.shape, sample.shape)
        element = torch.cat([element, element[sample]], dim=0)
        a = torch.rand(2, n_points, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x=a[1]*(a[0]*x1+(1-a[0])*x2)+(1-a[1])*x0
        return x

    def subdivide_elements(self):
         self.node, self.element = TriangleMesh.subdivide(self.node, self.element)

    @staticmethod
    def subdivide(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #     x2
        #    /  \
        #   x5-- x4
        #  / \  / \
        # x0--x3--x1
        x3=(x0+x1)/2
        x4=(x1+x2)/2
        x5=(x0+x2)/2
        '''
        #wrong: many copies of the same node
        node_new=torch.cat([x0, x1, x2, x3, x4, x5], dim=0)
        idx0=torch.arange(0, len(element)).view(-1,1)
        idx1=idx0+len(element)
        idx2=idx1+len(element)
        idx3=idx2+len(element)
        idx4=idx3+len(element)
        idx5=idx4+len(element)
        element_new0=torch.cat([idx2, idx5, idx4], dim=1)
        element_new1=torch.cat([idx5, idx0, idx3], dim=1)
        element_new2=torch.cat([idx5, idx3, idx4], dim=1)
        element_new3=torch.cat([idx4, idx3, idx1], dim=1)
        element_new=torch.cat([element_new0, element_new1, element_new2, element_new3], dim=0)
        return node_new, element_new
        '''
        pass
#%%
if __name__ == "__main__":
    filename="C:/Research/MLFEA/TAVR/wall_tri.vtk"
    wall=TriangleMesh()
    wall.load_from_vtk(filename)
    wall.update_node_normal()
    wall.node+=wall.node_normal

    wall.save_by_vtk("C:/Research/MLFEA/TAVR/wall_tri_offset.vtk")

    wall.node, wall.element = TriangleMesh.subdivide(wall.node, wall.element)
    wall.save_by_vtk("C:/Research/MLFEA/TAVR/wall_tri_offset_sub.vtk")
    #%%
    wall.update_element_area()
    #%%
    points=wall.sample_points_on_elements(10*len(wall.node))
