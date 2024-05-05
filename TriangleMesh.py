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
from QuadMesh import QuadMesh
import PolygonMeshProcessing as pmp
#%%
class TriangleMesh(PolygonMesh):
    #3-node triangle element mesh
    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polygon_tri3'
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None
        if (node is not None) and (element is not None):
            if not self.is_tri():
                raise ValueError('not a triangle mesh')
        
    def update_node_normal(self, angle_weighted=True):
        self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element)
        self.node_normal=TriangleMesh.cal_node_normal(self.node, self.element, angle_weighted, self.element_normal)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ TriangleMesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True, element_normal=None):
        if element_normal is None:
            element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element)
        M=element.shape[0]
        e_normal=element_normal.view(M, 1, 3)
        e_normal=e_normal.expand(M, 3, 3)
        e_normal=e_normal.reshape(M*3, 3)
        N=node.shape[0]
        if angle_weighted == True:
            e_angle=TriangleMesh.cal_element_corner_angle(node, element)#e_angle: (M,3)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            e_normal=e_normal*weight.view(M*3,1)
        node_normal = torch_scatter.scatter(e_normal, element.view(-1), dim=0, dim_size=N, reduce="sum")        
        normal_norm=norm(node_normal, ord=2, dim=-1, keepdim=True)
        with torch.no_grad():
            normal_norm.data.clamp_(min=1e-12)        
        node_normal=node_normal/normal_norm
        node_normal=node_normal.contiguous()
        return node_normal

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=TriangleMesh.cal_element_area_and_normal(self.node, self.element)

    def update_element_corner_angle(self):
        self.element_corner_angle = TriangleMesh.cal_element_corner_angle(self.node, self.element)

    @staticmethod
    def cal_element_area_and_normal(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #   x2
        #  /  \
        # x0--x1
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
        element_area, element_normal=TriangleMesh.cal_element_area_and_normal(node, element)
        return element_normal

    @staticmethod
    def cal_element_corner_angle(node, element, return_cos=False):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        #   x2
        #  /  \
        # x0--x1
        angle0=pmp.ComputeAngleBetweenTwoVectorIn3D(x1-x0, x2-x0, return_cos)
        angle1=pmp.ComputeAngleBetweenTwoVectorIn3D(x2-x1, x0-x1, return_cos)
        angle2=pmp.ComputeAngleBetweenTwoVectorIn3D(x0-x2, x1-x2, return_cos)
        angle=torch.cat([angle0.view(-1,1), angle1.view(-1,1), angle2.view(-1,1)], dim=1)
        return angle

    def sample_points_on_elements(self, n_points):
        return TriangleMesh.sample_points(self.node, self.element, n_points)

    @staticmethod
    def sample_points(node, element, n_points):
        area, normal=TriangleMesh.cal_element_area_and_normal(node, element)
        prob = area / area.sum()
        sample = torch.multinomial(prob.view(-1), n_points-len(element), replacement=True)
        #print("sample_points", area.shape, prob.shape, sample.shape)
        element = torch.cat([element, element[sample]], dim=0)
        a = torch.rand(2, n_points, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x=a[1]*(a[0]*x1+(1-a[0])*x2)+(1-a[1])*x0
        return x

    def subdivid_to_tri_1to4(self):
        #return a new mesh
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
        element=self.element.tolist()
        element_new=[]
        for m in range(0, element.shape[0]):
            #-----------
            #     x2
            #    /  \
            #   x5-- x4
            #  / \  / \
            # x0--x3--x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)    
            element_new.append([id0, id3, id5])
            element_new.append([id3, id4, id5])
            element_new.append([id3, id1, id4])
            element_new.append([id5, id4, id2])
        mesh_new=TriangleMesh(node_new, element_new)
        return mesh_new
    
    def subdivid_to_tri_1to3(self, mode):
        #return a new mesh
        #add a node in the middle of each face        
        nodeA=self.node[self.element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA], dim=0)        
        element=self.element.tolist()
        element_new=[]
        for m in range(0, len(element)):
            #-----------
            #     x2
            #    / | \
            #   / x3  \
            #  / /  \  \
            #  x0-----x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+m
            element_new.append([id0, id1, id3])
            element_new.append([id0, id3, id2])
            element_new.append([id1, id2, id3])
        mesh_new=TriangleMesh(node_new, element_new)
        return mesh_new    

    def subdivid_to_tri(self, mode):
        if mode == "1to4":
            return self.subdivid_to_tri_1to4()
        elif mode == "1to3":
            return self.subdivid_to_tri_1to3()
        else:
            raise ValueError('unsupported mode: '+str(mode))
        
    def subdivide_to_quad(self):
        #return a new mesh
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2
        #add a node in the middle of each element
        nodeB=self.node[self.element].mean(dim=1)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element=self.element.tolist()
        element_new=[]
        for m in range(0, element.shape[0]):
            #-----------
            #     x2
            #     /\
            #   x5  x4
            #   / \/ \
            #  /  x6  \
            # /   |    \
            #x0---x3---x1
            #-----------
            id0=element[m][0]
            id1=element[m][1]
            id2=element[m][2]
            id3=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id2)
            id6=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id6, id5, id0, id3])
            element_new.append([id6, id3, id1, id4])
            element_new.append([id6, id4, id2, id5])
        mesh_new=QuadMesh(node_new, element_new)
        return mesh_new

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        element_sub=self.element[element_idx_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_new=self.node[node_idx_list]
        element_new=element_out.view(len(element_idx_list),-1)
        mesh_new=TriangleMesh(node_new, element_new)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    #%%
    node=np.array([[-1, 0, 0],
                    [1, 0, 0],
                    [0, 2, 0]], dtype='float32')
    element=[[0,1,2]]
    tri_mesh=TriangleMesh(node, element)
    element_angle=TriangleMesh.cal_element_corner_angle(tri_mesh.node, tri_mesh.element)
    #%%
    mesh1=tri_mesh.subdivide_to_quad()
    for i in range(4):
        mesh1=mesh1.subdivide()
    mesh1.save_as_vtk("simple_tri.vtk")
    #%%
    mesh2=mesh1.get_sub_mesh([0, 3, 9, 1, 2])
    #%%
    mesh1p=PolygonMesh(mesh1.node, mesh1.element)
    mesh2p=mesh1p.get_sub_mesh([0, 3, 9, 1, 2])
    #%%
    filename="wall_tri.vtk"
    wall=TriangleMesh()
    wall.load_from_vtk(filename, 'float64')
    wall.update_node_normal()
    wall.node+=wall.node_normal
    wall.save_by_vtk("wall_tri_offset.vtk")
    wall_sub = wall.subdivide()
    wall_sub.save_by_vtk("wall_tri_offset_sub.vtk")
    #%%
    wall.update_element_area_and_normal()
    #%%
    points=wall.sample_points_on_elements(10*len(wall.node))
    #%%
    wall_sub=wall.get_sub_mesh(torch.arange(0,100))
    wall_sub.save_by_vtk("wall_tri_sub.vtk")
