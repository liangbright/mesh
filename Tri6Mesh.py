# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 18:45:59 2024

@author: liang
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from PolygonMesh import PolygonMesh
from TriangleMesh import TriangleMesh
import PolygonMeshProcessing as pmp
#%%
class Tri6Mesh(PolygonMesh):
    #6-node triangle element mesh
    #     x2
    #    /  \
    #   x5   x4
    #  /      \
    # x0--x3--x1
    # node order in an element: [x0, x1, x2, x3, x4, x5]
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polygon_tri6'
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None
        #if (node is not None) and (element is not None):
        #    if not self.is_tri6():
        #        raise ValueError('not a tri6 mesh')
    
    def create_from_tri3_mesh(self, tri3_mesh):
        #tri3_mesh is a TriangleMesh (3 nodes in an element)
        #add a node in the middle of each edge of tri3_mesh
        if tri3_mesh.edge is None:
            tri3_mesh.build_edge()
        x_j=tri3_mesh.node[tri3_mesh.edge[:,0]]
        x_i=tri3_mesh.node[tri3_mesh.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([tri3_mesh.node, nodeA], dim=0)        
        element=tri3_mesh.element.tolist()
        element_new=[]
        for m in range(0, element.shape[0]):
            #-----------
            #     x2
            #    /  \
            #   x5  x4
            #  /      \
            # x0--x3--x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id0, id1)
            id4=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id1, id2)
            id5=tri3_mesh.node.shape[0]+tri3_mesh.get_edge_idx_from_node_pair(id0, id2)    
            element_new.append([id0, id1, id2, id3, id4, id5])
        self.node=node_new
        self.element=torch.tensor(element_new, dtype=torch.int64)
    
    def update_node_normal(self, angle_weighted=True):
        self.element_area, self.element_normal=Tri6Mesh.cal_element_area_and_normal(self.node, self.element)
        self.node_normal=Tri6Mesh.cal_node_normal(self.node, self.element, angle_weighted, self.element_normal)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ Tri6Mesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, element_normal=None):
        if element_normal is None:
            element_area, element_normal=Tri6Mesh.cal_element_area_and_normal(node, element)
        M=element.shape[0]
        e_normal=element_normal.view(M, 1, 3)
        e_normal=e_normal.expand(M, 6, 3)
        e_normal=e_normal.reshape(M*6, 3)
        N=node.shape[0]
        if angle_weighted == True:
            e_angle=Tri6Mesh.cal_element_corner_angle(node, element)#e_angle: (M,6)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            e_normal=e_normal*weight.view(M*6,1)
        node_normal = torch_scatter.scatter(e_normal, element.view(-1), dim=0, dim_size=N, reduce="sum")        
        normal_norm=norm(node_normal, ord=2, dim=-1, keepdim=True)
        with torch.no_grad():
            normal_norm.data.clamp_(min=1e-12)        
        node_normal=node_normal/normal_norm
        node_normal=node_normal.contiguous()
        return node_normal

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=Tri6Mesh.cal_element_area_and_normal(self.node, self.element)

    def update_element_corner_angle(self):
        self.element_corner_angle = Tri6Mesh.cal_element_corner_angle(self.node, self.element)

    @staticmethod
    def cal_element_area_and_normal(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #     x2
        #    /  \
        #   x5   x4
        #  /      \
        # x0--x3--x1
        #normal is undefined if area is 0
        temp1=torch.cross(x1-x0, x2-x0, dim=-1)        
        temp2=norm(temp1, ord=2, dim=-1, keepdim=True)
        area=0.5*temp2.abs()
        with torch.no_grad():
            #https://github.com/pytorch/pytorch/issues/43211
            temp2.data.clamp_(min=1e-12)
        normal=temp1/temp2
        return area, normal

    @staticmethod
    def cal_element_normal(node, element):
        element_area, element_normal=Tri6Mesh.cal_element_area_and_normal(node, element)
        return element_normal

    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x4=node[element[:,4]]
        x5=node[element[:,5]]
        #     x2
        #    /  \
        #   x5   x4
        #  /      \
        # x0--x3--x1
        angle0=pmp.ComputeAngleBetweenTwoVectorIn3D(x3-x0, x5-x0, return_cos)
        angle1=pmp.ComputeAngleBetweenTwoVectorIn3D(x4-x1, x3-x1, return_cos)
        angle2=pmp.ComputeAngleBetweenTwoVectorIn3D(x5-x2, x4-x2, return_cos)
        angle3=pmp.ComputeAngleBetweenTwoVectorIn3D(x1-x3, x0-x3, return_cos)
        angle4=pmp.ComputeAngleBetweenTwoVectorIn3D(x2-x4, x1-x4, return_cos)
        angle5=pmp.ComputeAngleBetweenTwoVectorIn3D(x0-x5, x2-x5, return_cos)
        angle=torch.cat([angle0.view(-1,1), angle1.view(-1,1), angle2.view(-1,1),
                         angle3.view(-1,1), angle4.view(-1,1), angle5.view(-1,1)], dim=1)
        return angle

    def get_tri3_element(self):
        tri3_element=[]
        for m in len(self.element):
            tri3_element.append([int(self.element[m,0]), int(self.element[m,1]), int(self.element[m,2])])    
        tri3_element=torch.tensor(tri3_element, dtype=torch.int64)
        return tri3_element

    def convert_to_tri3_mesh(self, return_node_idx_list=False):
        element_sub=self.get_tri3_element()
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_new=self.node[node_idx_list]
        element_new=element_out.view(len(element_sub),-1)
        mesh_new=TriangleMesh(node_new, element_new)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        element_sub=self.element[element_idx_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_new=self.node[node_idx_list]
        element_new=element_out.view(len(element_idx_list),-1)
        mesh_new=Tri6Mesh(node_new, element_new)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    #%%
    pass