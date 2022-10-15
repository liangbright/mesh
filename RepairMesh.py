# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:32:29 2022

@author: liang
"""
import numpy as np
from PolygonMesh import PolygonMesh
from copy import deepcopy
#%%
def ReordeNodeInElement(mesh, ref_element_id, verbose=False):
    #mesh is PolygonMesh
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #
    mesh.build_element_to_element_table(adj=2)
    good_element_list=[ref_element_id]
    active_element_list=[ref_element_id]
    while True:
        next_active_element_list=[]
        for elm_id in active_element_list:
            adj_elm_idxlist=mesh.element_to_element_table[elm_id]
            adj_elm_idxlist=list(set(adj_elm_idxlist)-set(good_element_list))
            next_active_element_list.extend(adj_elm_idxlist)
            for adj_elm_id in adj_elm_idxlist:
                shared_node_id=np.intersect1d(mesh.element[elm_id], mesh.element[adj_elm_id])
                if len(shared_node_id) != 2:
                    raise NotImplementedError("not repairable")
                idx0=np.where(np.array(mesh.element[elm_id])==shared_node_id[0])[0].item()
                idx1=np.where(np.array(mesh.element[elm_id])==shared_node_id[1])[0].item()
                idx0a=np.where(np.array(mesh.element[adj_elm_id])==shared_node_id[0])[0].item()
                idx1a=np.where(np.array(mesh.element[adj_elm_id])==shared_node_id[1])[0].item()
                if idx0 == 0 and idx1 == len(mesh.element[elm_id])-1:
                    direction="1to0"
                elif idx1 == 0 and idx0 == len(mesh.element[elm_id])-1:
                    direction="0to1"
                elif idx0 < idx1:
                    direction="0to1"
                elif idx1 < idx0:
                    direction="1to0"
                if idx0a == 0 and idx1a == len(mesh.element[adj_elm_id])-1:
                    direction_a="1to0"
                elif idx1a == 0 and idx0a == len(mesh.element[adj_elm_id])-1:
                    direction_a="0to1"
                elif idx0a < idx1a:
                    direction_a="0to1"
                elif idx1a < idx0a:
                    direction_a="1to0"
                if direction == direction_a:
                    #reverse node order in adj_elm
                    adj_elm=mesh.element[adj_elm_id]
                    mesh.element[adj_elm_id]=adj_elm[-1::-1]
                    if verbose == True:
                        print("reverse node order in adj_elm", adj_elm_id, adj_elm, adj_elm[-1::-1])
            good_element_list.extend(adj_elm_idxlist)
        next_active_element_list=list(np.unique(next_active_element_list))
        active_element_list=next_active_element_list
        if len(active_element_list) == 0:
            break
#%%
def SmoothMesh(mesh, node_idxlist=None, element_idxlist=None):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    mesh.build_node_to_node_table()
    node_idxlist_input=[]
    if node_idxlist is not None:
        node_idxlist_input=deepcopy(node_idxlist)
    if element_idxlist is not None:
        for idx in element_idxlist:
            node_idxlist_input.extend(mesh.element[idx])
    for idx in node_idxlist_input:
        adj_node_idxlist=mesh.node_to_node_table[idx]
        adj_node_idxlist.append(idx)
        avg=mesh.node[adj_node_idxlist].mean(dim=0)
        mesh.node[idx]=avg
#%%
if __name__ == "__main__":
    #%%
    filename="E:/TAA_dataset10/INP_new/P1t2_surface.pt"
    mesh=PolygonMesh()
    mesh.load_from_torch(filename)
    #
    ReordeNodeInElement(mesh, 0)
    mesh.save_by_vtk("E:/TAA_dataset10/INP_new/P1t2_surface_fixed.vtk")
    #%%
    filename="E:/TAA_dataset10/INP_new/abaqus_error/P1t3_surface.pt"
    mesh=PolygonMesh()
    mesh.load_from_torch(filename)
    SmoothMesh(mesh, None, [10197, 10237, 10796, 12025, 12781])
    mesh.save_by_vtk("E:/TAA_dataset10/INP_new/abaqus_error/P1t3_surface_fixed.vtk")



