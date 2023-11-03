# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:50:36 2023

@author: liang
"""
#%%
import torch
import torch_scatter
import numpy as np
from PolygonMesh import PolygonMesh
from TriangleMesh import TriangleMesh
from QuadMesh import QuadMesh
from QuadTriangleMesh import QuadTriangleMesh
from copy import deepcopy
from MeshProcessing import SimpleSmoother, SimpleSmootherForMesh, ComputeAngleBetweenTwoVectorIn3D, TracePolyline, \
                           IsCurveClosed, MergeMesh, FindConnectedRegion
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
    if mesh.node_to_node_adj_table is None:
        mesh.build_node_to_node_adj_table()
    node_adj_table=mesh.node_to_node_adj_table
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
def ExtractRegionEnclosedByCurve(mesh, node_curve_list, inner_element_id):
    #node_curve_list[k] is a curve - represented by a list/array of node indexes on mesh
    #the combined curve (from curve_list[0] to curve_list[-1]) is closed
    #inner_element_id is the index of an element inside the region
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #-------------------------
    curve=[]
    for k in range(0, len(node_curve_list)):
        curve_k=[int(x) for x in node_curve_list[k]]
        curve.extend(curve_k)
    if not IsCurveClosed(mesh, curve):
        raise ValueError('curve is not closed')
    #-------------------------
    if mesh.edge_to_element_adj_table is None:
        mesh.build_edge_to_element_adj_table()
    edge_to_element_adj_table=mesh.edge_to_element_adj_table
    #-------------------------
    edge_curve=[]
    for k in range(0, len(node_curve_list)):
        curve_k=[int(x) for x in node_curve_list[k]]
        for n in range(0, len(curve_k)):
            idx_n=int(curve_k[n])
            if n < len(curve_k)-1:
                idx_n1=int(curve_k[n+1])
            else:
                idx_n1=int(curve_k[0])
            edge_idx=mesh.get_edge_idx_from_node_pair(idx_n, idx_n1)
            edge_curve.append(edge_idx)
    #-------------------------
    region_element_list=[inner_element_id]
    active_element_list=[inner_element_id]
    counter=0
    while True:
        new_active_element_list=[]
        region_element_set=set(region_element_list)
        active_element_set=set(active_element_list)
        for act_elm_idx in active_element_list:
            for n in range(0, len(mesh.element[act_elm_idx])):
                idx_n=int(mesh.element[act_elm_idx][n])
                if n < len(mesh.element[act_elm_idx])-1:
                    idx_n1=int(mesh.element[act_elm_idx][n+1])
                else:
                    idx_n1=int(mesh.element[act_elm_idx][0])
                edge_idx=mesh.get_edge_idx_from_node_pair(idx_n, idx_n1)
                if edge_idx not in edge_curve:
                    adj_elm_idx=edge_to_element_adj_table[edge_idx]
                    adj_elm_idx=list(set(adj_elm_idx)-set(active_element_set))
                    new_active_element_list.extend(adj_elm_idx)
        if len(new_active_element_list) == 0:
            break
        active_element_list=list(set(new_active_element_list)-region_element_set)
        region_element_list.extend(active_element_list)
        counter+=1
        #if counter ==100:
        #    break
    #--------------
    return region_element_list
#%%
def SimpleSmootherForMeshNodeNormal(mesh, lamda, mask, n_iters):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    mesh.update_node_normal()
    node_normal=mesh.node_normal
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link
    for n in range(0, n_iters):
        SimpleSmoother(node_normal, adj_link, lamda, mask, inplace=True)
        normal_norm=torch.norm(node_normal, p=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        node_normal=node_normal/normal_norm
    node_normal=node_normal.contiguous()
    mesh.node_normal=node_normal
#%%
def MergeMeshOnBoundary(mesh_list, distance_threshold):
    for mesh in mesh_list:
        if not isinstance(mesh, PolygonMesh):
            raise NotImplementedError
    merged_mesh=mesh_list[0]
    for n in range(1, len(mesh_list)):
        mesh_n=mesh_list[n]
        merged_mesh= MergeMesh(merged_mesh, merged_mesh.find_boundary_node(),
                               mesh_n, mesh_n.find_boundary_node(),
                               distance_threshold)
    return merged_mesh
