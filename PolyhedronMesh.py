import torch
import numpy as np
from Mesh import Mesh
#%%
class PolyhedronMesh(Mesh):
    
    def __init__(self, node=None, element=None, element_type=None):
        super().__init__(node=node, element=element, element_type=element_type, mesh_type='polyhedron')
    
    def build_face_to_element_adj_table(self):
        if self.element_to_face_adj_table is None:
            self.build_element_to_face_adj_table()
        adj_table=[[] for _ in range(len(self.face))]
        for m in range(0, len(self.element)):
            face_idx_list=self.element_to_face_adj_table[m]
            for idx in face_idx_list:
                adj_table[idx].append(m)
        self.face_to_element_adj_table=adj_table

    def find_boundary_face(self):
        if self.face_to_element_adj_table is None:
            self.build_face_to_element_adj_table()
        face_idx_list=[]
        for k in range(0, len(self.face)):
            adj_elm_idx=self.face_to_element_adj_table[k]
            if len(adj_elm_idx) == 1:
                face_idx_list.append(k)
        return face_idx_list

    def find_boundary_node(self):
        face_idx_list=self.find_boundary_face()
        node_idx_list=[]
        for idx in face_idx_list:
            node_idx_list.extend(self.face[idx])
        node_idx_list=np.unique(node_idx_list).tolist()
        return node_idx_list
    
    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        new_mesh, node_idx_list=super().get_sub_mesh(element_idx_list, return_node_idx_list=True)
        new_mesh=PolyhedronMesh(new_mesh.node, new_mesh.element)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
#%%
if __name__ == "__main__":
    #
    root=PolyhedronMesh()
    root.load_from_vtk("D:/MLFEA/TAA/data/343c1.5/matMean/p0_0_solid_matMean_p20.vtk", 'float64')
    root.save_by_vtk("D:/MLFEA/TAA/test_PolyhedronMesh.vtk")
