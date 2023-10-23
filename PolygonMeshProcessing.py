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
from copy import deepcopy
#%%
def ComputeAngleBetweenTwoVectorIn3D(VectorA, VectorB):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    eps = 1e-8
    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = np.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if np.any(L2Norm_A <= eps) or np.any(L2Norm_B <= eps):
            print("L2Norm < eps, return 0 @ ComputeAngleBetweenTwoVectorIn3D(...)")
            return 0
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, min=-1, max=1)
        Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA,  torch.Tensor):
        L2Norm_A = torch.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = torch.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if torch.any(L2Norm_A <= eps) or torch.any(L2Norm_B <= eps):
            print("L2Norm < eps, return 0 @ ComputeAngleBetweenTwoVectorIn3D(...)")
            return 0
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = torch.clamp(CosTheta, min=-1, max=1)
        Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
    return Theta
#%%
def FindConnectedRegion(mesh, ref_element_idx, adj=2):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError

    mesh.build_element_adj_table(adj=adj)
    element_adj_table=mesh.element_adj_table["adj"+str(int(adj))]
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
#%%
def SimpleMeshSmoother(mesh, lamda, inplace, mask=None):
    #lamda: x_i = x_i + lamda*mean_j(x_j - x_i),  0<=lamda<=1
    #inplace: True or False
    #mask[k]: 1 to smooth the node-k; 0 not to smooth the node-k
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    if mask is None:
        mask=1
    if isinstance(mask, list):
        mask=torch.tensor(mask, dtype=mesh.node.dtype, device=mesh.node.device)
        mask=mask.view(-1,1)
    elif isinstance(mask, np.ndarray):
        mask=torch.tensor(mask, dtype=mesh.node.dtype, device=mesh.node.device)
        mask=mask.view(-1,1)
    elif isinstance(mask, torch.Tensor):
        mask=mask.to(mesh.node.dtype).to(mesh.node.device)
        mask=mask.view(-1,1)
    #---------
    if mesh.node_adj_link is None:
        mesh.build_node_adj_link()
    x_j=mesh.node[mesh.node_adj_link[:,0]]
    x_i=mesh.node[mesh.node_adj_link[:,1]]
    delta=x_j-x_i
    delta=torch_scatter.scatter(delta, mesh.node_adj_link[:,1], dim=0, dim_size=mesh.node.shape[0], reduce="mean")
    delta=lamda*delta*mask
    if inplace == True:
        mesh.node+=delta
        return mesh
    else:
        new_node=mesh.node+delta
        new_element=deepcopy(mesh.element)
        new_mesh=PolygonMesh(new_node, new_element)
        return new_mesh



