# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
from Mesh import Mesh
#%%
class PolyhedronMesh(Mesh):
    # use this class to handle mixture of tetra and hex elements
    def __init__(self, node=None, element=None, dtype=torch.float32):
        super().__init__('polyhedron')

        if node is not None:
            if isinstance(node, list):
                node=torch.tensor(node, dtype=dtype)
            elif isinstance(node, np.ndarray):
                node=torch.tensor(node, dtype=dtype)
            elif isinstance(node,  torch.Tensor):
                pass
            else:
                raise ValueError("unkown data type of node")
            self.node=node

        if element is not None:
            if isinstance(element, list) or isinstance(node, np.ndarray):
                try:
                    element=torch.tensor(element, dtype=torch.int64)
                except:
                    pass
            elif isinstance(element,  torch.Tensor):
                  pass
            else:
                raise ValueError("unkown data type of element")
            self.element=element

    def get_sub_mesh(self, element_idx_list):
        sub_mesh=super().get_sub_mesh(element_idx_list)
        new_mesh=PolyhedronMesh()
        new_mesh.node=sub_mesh.node
        new_mesh.element=sub_mesh.element
        return new_mesh
#%%
if __name__ == "__main__":
    #
    root=PolyhedronMesh()
    root.load_from_vtk("D:/MLFEA/TAA/data/343c1.5/matMean/p0_0_solid_matMean_p20.vtk", 'float64')
    root.save_by_vtk("D:/MLFEA/TAA/test_PolyhedronMesh.vtk")
