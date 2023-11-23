# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import torch_scatter
from PolygonMesh import PolygonMesh
import PolygonMeshProcessing as pmp
#%%
class QuadMesh(PolygonMesh):
    #4-node quad element mesh

    def __init__(self, node=None, element=None, dtype=None):
        super().__init__(node=node, element=element, dtype=dtype)
        self.mesh_type='polygon_quad4'
        self.node_normal=None
        self.element_area=None
        self.element_normal=None
        self.element_corner_angle=None

    def update_node_normal(self, angle_weighted=True):
        self.node_normal=QuadMesh.cal_node_normal(self.node, self.element, angle_weighted=angle_weighted)
        error=torch.isnan(self.node_normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ TriangleMesh:update_node_normal")

    @staticmethod
    def cal_node_normal(node, element, angle_weighted=True):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        #normal could be nan if xa==xb (e.g., x0 == x1)
        normal0=torch.cross(x1-x0, x3-x0, dim=-1)
        normal0_norm=torch.norm(normal0, p=2, dim=1, keepdim=True)
        with torch.no_grad():
            normal0_norm.data.clamp_(min=1e-12)            
        normal0=normal0/normal0_norm
        
        normal1=torch.cross(x2-x1, x0-x1, dim=-1)
        normal1_norm=torch.norm(normal1, p=2, dim=1, keepdim=True)
        with torch.no_grad():
            normal1_norm.data.clamp_(min=1e-12) 
        normal1=normal1/normal1_norm
        
        normal2=torch.cross(x3-x2, x1-x2, dim=-1)
        normal2_norm=torch.norm(normal2, p=2, dim=1, keepdim=True)
        with torch.no_grad():
            normal2_norm.data.clamp_(min=1e-12) 
        normal2=normal2/normal2_norm

        normal3=torch.cross(x0-x3, x2-x3, dim=-1)
        normal3_norm=torch.norm(normal3, p=2, dim=1, keepdim=True)
        with torch.no_grad():
            normal3_norm.data.clamp_(min=1e-12) 
        normal3=normal3/normal3_norm

        M=element.shape[0]
        N=node.shape[0]
        normal0123=torch.cat([normal0.view(M,1,3),
                              normal1.view(M,1,3),
                              normal2.view(M,1,3),
                              normal3.view(M,1,3)], dim=1)
        if angle_weighted == True:
            e_angle=QuadMesh.cal_element_corner_angle(node, element)#e_angle: (M,3)
            weight=e_angle/e_angle.sum(dim=1, keepdim=True)
            normal0123=normal0123*weight.view(M,4,1)
        normal=torch_scatter.scatter(normal0123.view(-1,3), element.view(-1), dim=0, dim_size=N, reduce="sum")
        error=torch.isnan(normal).sum()
        if error > 0:
            print("error: nan in normal_quad @ QuadMesh:cal_node_normal")
        normal_norm=torch.norm(normal, p=2, dim=1, keepdim=True)
        with torch.no_grad():
            normal_norm.data.clamp_(min=1e-12)
        normal=normal/normal_norm
        normal=normal.contiguous()
        return normal

    def update_element_area_and_normal(self):
         self.element_area, self.element_normal=QuadMesh.cal_element_area_and_normal(self.node, self.element)

    def update_element_corner_angle(self):
        self.element_corner_angle = QuadMesh.cal_element_corner_angle(self.node, self.element)

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
        cross_uv=torch.cross(dxdu, dxdv, dim=-1)
        temp=torch.norm(cross_uv, p=2, dim=1, keepdim=True)
        area=4*temp.abs()
        with torch.no_grad():
            temp.data.clamp_(min=1e-12)
        normal=cross_uv/temp
        return area, normal

    @staticmethod
    def cal_element_corner_angle(node, element):
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        # x3--x2
        # |   |
        # x0--x1
        angle0=pmp.ComputeAngleBetweenTwoVectorIn3D(x1-x0, x3-x0)
        angle1=pmp.ComputeAngleBetweenTwoVectorIn3D(x2-x1, x0-x1)
        angle2=pmp.ComputeAngleBetweenTwoVectorIn3D(x3-x2, x1-x2)
        angle3=pmp.ComputeAngleBetweenTwoVectorIn3D(x0-x3, x2-x3)
        angle=torch.cat([angle0.view(-1,1), angle1.view(-1,1), angle2.view(-1,1), angle3.view(-1,1)], dim=1)
        return angle

    def sample_points_on_elements(self, n_points):
         return QuadMesh.sample_points(self.node, self.element, n_points)

    @staticmethod
    def sample_points(node, element, n_points):
        area, normal=QuadMesh.cal_element_area_and_normal(node, element)
        prob = area / area.sum()
        sample = torch.multinomial(prob.view(-1), n_points-len(element), replacement=True)
        #print("sample_points", element.shape, area.shape, prob.shape, sample.shape)
        element = torch.cat([element, element[sample]], dim=0)
        a = torch.rand(3, n_points, 1, dtype=node.dtype, device=node.device)
        x0=node[element[:,0]]
        x1=node[element[:,1]]
        x2=node[element[:,2]]
        x3=node[element[:,3]]
        x=a[2]*(a[0]*x0+(1-a[0])*x1)+(1-a[2])*(a[1]*x2+(1-a[1])*x3)
        return x

    def subdivide(self):
        #return a new mesh        
        #add a node in the middle of each edge
        if self.edge is None:
            self.build_edge()
        x_j=self.node[self.edge[:,0]]
        x_i=self.node[self.edge[:,1]]
        nodeA=(x_j+x_i)/2        
        #add a node in the middle of each element
        nodeB=self.node[self.element].mean(dim=1) #(N,3) => (M,8,3) => (M,3)
        #create new mesh
        node_new=torch.cat([self.node, nodeA, nodeB], dim=0)
        element_new=[]
        if torch.is_tensor(self.element):
            element=self.element.cpu().numpy()
        for m in range(0, element.shape[0]):
            # x3--x6--x2
            # |   |   |
            # x7--x8--x5
            # |   |   |
            # x0--x4--x1
            #-----------
            id0=int(element[m][0])
            id1=int(element[m][1])
            id2=int(element[m][2])
            id3=int(element[m][3])
            id4=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id1)
            id5=self.node.shape[0]+self.get_edge_idx_from_node_pair(id1, id2)
            id6=self.node.shape[0]+self.get_edge_idx_from_node_pair(id2, id3)
            id7=self.node.shape[0]+self.get_edge_idx_from_node_pair(id0, id3)            
            id8=self.node.shape[0]+nodeA.shape[0]+m
            element_new.append([id0, id4, id8, id7])
            element_new.append([id4, id1, id5, id8])
            element_new.append([id7, id8, id6, id3])
            element_new.append([id8, id5, id2, id6])
        element_new=torch.tensor(element_new, dtype=torch.int64, device=self.element.device)
        mesh_new=QuadMesh(node_new, element_new)
        return mesh_new

    def get_sub_mesh(self, element_id_list, return_node_idx_list=False):
        element_sub=self.element[element_id_list]
        node_idx_list, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
        node_new=self.node[node_idx_list]
        element_new=element_out.view(-1,4)
        mesh_new=QuadMesh(node_new, element_new)
        if return_node_idx_list == False:
            return mesh_new
        else:
            return mesh_new, node_idx_list
#%%
if __name__ == "__main__":
    filename="F:/MLFEA/TAA/data/343c1.5/bav17_AortaModel_P0_best.vtk"
    wall=QuadMesh()
    wall.load_from_vtk(filename, 'float64')
    wall.update_node_normal()
    wall.node+=1.5*wall.node_normal
    wall.save_by_vtk("aorta_quad_offset.vtk")

    #wall.node, wall.element = QuadMesh.subdivide(wall.node, wall.element)
    #wall.save_to_vtk("C:/Research/MLFEA/TAVR/wall_quad_offset_sub.vtk")
    #%%
    wall.update_element_area_and_normal()
    #%%
    points=wall.sample_points_on_elements(10*len(wall.node))
    #%%
    wall_new=wall.subdivide()
    wall_new.save_by_vtk("wall_new.vtk")
    #%%
    sub_mesh=wall_new.get_sub_mesh(torch.arange(0,100))
    sub_mesh.save_by_vtk("sub_mesh.vtk")
    #%%
    wall.quad_to_tri()
    wall.save_by_vtk("wall_tri.vtk")

