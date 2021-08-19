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
_Flag_VTK_IMPORT_=False
try:
    import vtk
    from vtk import vtkPoints
    from vtk import vtkPolyData, vtkPolyDataReader, vtkPolyDataWriter
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class QuadMesh(PolygonMesh):

    def __init__(self):
        super().__init__()

    def update_node_normal(self):
        self.node_normal=QuadMesh.cal_node_normal(self.node, self.element)

    @staticmethod
    def cal_node_normal(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        normal0=torch.cross(x1-x0, x3-x0)
        normal0=normal0/torch.norm(normal0, p=2, dim=1, keepdim=True)
        normal1=torch.cross(x2-x1, x0-x1)
        normal1=normal1/torch.norm(normal1, p=2, dim=1, keepdim=True)
        normal2=torch.cross(x3-x2, x1-x2)
        normal2=normal2/torch.norm(normal2, p=2, dim=1, keepdim=True)
        normal3=torch.cross(x0-x3, x2-x3)
        normal3=normal3/torch.norm(normal3, p=2, dim=1, keepdim=True)
        M=element.shape[0]
        N=node.shape[0]
        normal0123=torch.cat([normal0.view(M,1,3),
                              normal1.view(M,1,3),
                              normal2.view(M,1,3),
                              normal3.view(M,1,3)], dim=1)
        normal=torch_scatter.scatter(normal0123.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
        normal_norm=torch.norm(normal, p=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        normal=normal/normal_norm
        normal=normal.contiguous()
        return normal

    def update_edge_length(self):
         self.edge_length=QuadMesh.cal_edge_length(self.node, self.adj_node_link)

    @staticmethod
    def cal_edge_length(node, adj_node_link):
        x_i=node[adj_node_link[:,0]]
        x_j=node[adj_node_link[:,1]]
        edge_length=torch.norm(x_j-x_i, p=2, dim=1, keepdim=True)
        return edge_length

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=QuadMesh.cal_element_area_and_normal(self.node, self.element)

    @staticmethod
    def cal_element_area_and_normal(node, element):
        #area is an estimation using 1 integration point
        #normal is at the center
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        dxdu=(1/4)*((x1+x2)-(x0+x3))
        dxdv=(1/4)*((x2+x3)-(x0+x1))
        cross_uv=torch.cross(dxdu, dxdv)
        temp=torch.norm(cross_uv, p=2, dim=1, keepdim=True)
        area=4*temp.abs()
        temp=temp.clamp(min=1e-12)
        normal=cross_uv/temp
        return area, normal

    def sample_points_on_elements(self, n_points):
         return QuadMesh.sample_points(self.node, self.element, n_points)

    @staticmethod
    def sample_points(node, element, n_points):
        area=QuadMesh.cal_element_area(node, element)
        prob = area / area.sum()
        sample = torch.multinomial(prob, n_points-len(element), replacement=True)
        #print("sample_points", area.shape, prob.shape, sample.shape)
        element = torch.cat([element, element[sample]], dim=0)
        a = torch.rand(3, n_points, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x=a[2]*(a[0]*x0+(1-a[0])*x1)+(1-a[2])*(a[1]*x2+(1-a[1])*x3)
        return x

    def subdivide_elements(self):
         self.node, self.element = QuadMesh.subdivide(self.node, self.element)

    @staticmethod
    def subdivide(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x6--x2
        # |   |   |
        # x7--x8--x5
        # |   |   |
        # x0--x4--x1
        x4=(x0+x1)/2
        x5=(x1+x2)/2
        x6=(x2+x3)/2
        x7=(x0+x3)/2
        x8=(x5+x7)/2
        pass
        #return node_new, element_new
#%%
def get_sub_mesh(node, element, element_sub):
    #element.shape (M,4)
    node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
    node_out=node[node_idlist]
    element_out=element_out.view(-1,4)
    return node_out, element_out
#%%
if __name__ == "__main__":
    filename="C:/Research/MLFEA/TAVR/aorta_quad.vtk"
    wall=QuadMesh()
    wall.load_from_vtk(filename)
    wall.update_node_normal()
    wall.node+=1.5*wall.node_normal

    wall.save_by_vtk("C:/Research/MLFEA/TAVR/aorta_quad_offset.vtk")

    #wall.node, wall.element = QuadMesh.subdivide(wall.node, wall.element)
    #wall.save_to_vtk("C:/Research/MLFEA/TAVR/wall_quad_offset_sub.vtk")
    #%%
    wall.update_element_area()
    #%%
    points=wall.sample_points_on_elements(10*len(wall.node))
