# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:50:36 2023

@author: liang
"""
#%%
import numpy as np
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
#%%
def TracePolygonMeshBoundaryCurve(mesh, node_idx):
    #trace boundary starting from node_ix
    #this function may not work well if two boundary curves share points
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    boundary=mesh.find_boundary_node()
    boundary=list(boundary)
    BoundaryCurve=[]
    if node_idx not in boundary:
        return BoundaryCurve
    #---------
    mesh.build_node_adj_table()
    node_adj_table=mesh.node_adj_table
    idx_next=node_idx
    while True:
        BoundaryCurve.append(idx_next)
        idx_list_next=node_adj_table[idx_next]
        flag=False
        for k in range(0, len(idx_list_next)):
            idx_next=idx_list_next[k]
            if (idx_next in boundary) and (idx_next not in BoundaryCurve):
                flag=True
                break
        if flag == False:
            break
    return BoundaryCurve
#%%
def IsCurveClosed(mesh, curve):
    #curve is list/array of node indexes on mesh
    #if curve is closed, then return True
    #the nodes in curve could be in a random order
    curve=[int(x) for x in curve]
    curve=set(curve)
    mesh.build_node_adj_table()
    node_adj_table=mesh.node_adj_table
    for idx in curve:
        adj_idx_list=node_adj_table[idx]
        temp=curve.intersection(set(adj_idx_list))
        if len(temp) < 2:
            return False
    return True
#%%
def ExtractRegionEnclosedByCurve(mesh, curve, inner_node_idx):
    #curve is list/array of node indexes on mesh
    #curve is closed and has no self-intersection
    #inner_node_idx is the index of a node inside the region
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    if not IsCurveClosed(mesh, curve):
        raise ValueError('curve is not closed')

    curve=[int(x) for x in curve]
    curve=set(curve)

    #element=mesh.element this will lead to error
    element=mesh.copy_element('list')

    mesh.build_element_adj_table(adj=1)
    element_adj_table=mesh.element_adj_table['adj1']
    mesh.build_node_to_element_adj_table()
    node_to_element_adj_table=mesh.node_to_element_adj_table

    inner_element_list=[]
    active_element_list=[]
    adj_elm_idx_list=node_to_element_adj_table[inner_node_idx]
    #TODO: check if inner_node_idx is really inside region
    inner_element_list.extend(adj_elm_idx_list)
    for adj_elm_idx in adj_elm_idx_list:
        if curve.isdisjoint(element[adj_elm_idx]):
            active_element_list.append(adj_elm_idx)

    counter=0
    while True:
        new_active_element_list=[]
        inner_element_set=set(inner_element_list)
        for act_elm_idx in active_element_list:
            adj_elm_idx_list=element_adj_table[act_elm_idx]
            adj_elm_idx_list=list(set(adj_elm_idx_list)-inner_element_set)
            for adj_elm_idx in adj_elm_idx_list:
                if curve.isdisjoint(element[adj_elm_idx]):
                    new_active_element_list.append(adj_elm_idx)
        if len(new_active_element_list) == 0:
            break
        new_active_element_list=np.unique(new_active_element_list).tolist()
        active_element_list=new_active_element_list
        inner_element_list.extend(active_element_list)
        counter+=1
        #if counter ==100:
        #    break
    #--------------
    inner_node_list=[]
    for elm_idx in inner_element_list:
        inner_node_list.extend(element[elm_idx])
    inner_node_list=np.unique(inner_node_list).tolist()
    #--------------
    region_element_list=[]
    for idx in inner_node_list:
        region_element_list.extend(node_to_element_adj_table[idx])
    region_element_list=np.unique(region_element_list).tolist()
    return region_element_list



