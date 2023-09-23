# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:50:36 2023

@author: liang
"""
#%%
from PolygonMesh import PolygonMesh
#%%
def FindConnectedRegion(mesh, ref_element_idx, adj=2):
    #mesh is PolygonMesh
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError

    mesh.build_element_adj_table(adj=adj)
    element_adj_table=mesh.element_adj_table["adj2"]
    region_element_list=[ref_element_idx]
    active_element_list=[ref_element_idx]
    while True:
        next_active_element_list=[]
        for elm_idx in active_element_list:
            adj_elm_idxlist=element_adj_table[elm_idx]
            adj_elm_idxlist=list(set(adj_elm_idxlist)-set(region_element_list))
            next_active_element_list.extend(adj_elm_idxlist)
            region_element_list.extend(adj_elm_idxlist)
    return region_element_list
