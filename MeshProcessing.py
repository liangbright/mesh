# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:12:58 2023

@author: liang
"""
import torch
import torch_scatter
import numpy as np
import Mesh
from copy import deepcopy
#%%
def ComputeAngleBetweenTwoVectorIn3D(VectorA, VectorB, eps = 1e-8):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.view(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.view(1,3)

    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = np.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if np.any(L2Norm_A <= eps) or np.any(L2Norm_B <= eps):
            print("L2Norm <= eps, np.clip to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
            L2Norm_A=np.clip(L2Norm_A, min=eps)
            L2Norm_B=np.clip(L2Norm_B, min=eps)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, min=-1, max=1)
        Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA,  torch.Tensor):
        L2Norm_A = torch.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = torch.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if torch.any(L2Norm_A <= eps) or torch.any(L2Norm_B <= eps):
            print("L2Norm <= eps, torch.clamp to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
            L2Norm_A=torch.clamp(L2Norm_A, min=eps)
            L2Norm_B=torch.clamp(L2Norm_B, min=eps)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = torch.clamp(CosTheta, min=-1, max=1)
        Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
    return Theta
#%%
def FindConnectedRegion(mesh, ref_element_idx, adj):
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    if mesh.element_to_element_adj_table[adj] is None:
        mesh.build_element_to_element_adj_table(adj=adj)
    element_adj_table=mesh.element_to_element_adj_table[adj]
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
def SimpleSmoother(field, adj_link, lamda, mask, inplace):
    #field.shape (N, ?)
    if mask is None:
        mask=1
    else:
        if isinstance(mask, list):
            mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
            mask=mask.view(-1,1)
        elif isinstance(mask, np.ndarray):
            mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
            mask=mask.view(-1,1)
        elif isinstance(mask, torch.Tensor):
            mask=mask.to(field.dtype).to(field.device)
            mask=mask.view(-1,1)
        else:
            raise ValueError('python-object type of mask is not supported')
    #---------
    x_j=field[adj_link[:,0]]
    x_i=field[adj_link[:,1]]
    delta=x_j-x_i
    delta=torch_scatter.scatter(delta, adj_link[:,1], dim=0, dim_size=field.shape[0], reduce="mean")
    delta=lamda*delta*mask
    if inplace == True:
        field+=delta
    else:
        field=field+delta
    return field
#%%
def SimpleSmootherForMesh(mesh, lamda, mask, n_iters):
    #lamda: x_i = x_i + lamda*mean_j(x_j - x_i),  0<=lamda<=1
    #inplace: True or False
    #if mask is not None: mask[k]: 1 to smooth the node-k; 0 not to smooth the node-k
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    #---------
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link
    for n in range(0, n_iters):
        SimpleSmoother(mesh.node, adj_link, lamda, mask, inplace=True)
#%%
def IsCurveClosed(mesh, curve):
    #curve is list/array of node indexes on mesh
    #if curve is closed, then return True
    #the nodes in curve could be in a random order
    curve=[int(x) for x in curve]
    curve=set(curve)
    if mesh.node_to_node_adj_table is None:
        mesh.build_node_to_node_adj_table()
    node_adj_table=mesh.node_to_node_adj_table
    for idx in curve:
        adj_idx_list=node_adj_table[idx]
        temp=curve.intersection(set(adj_idx_list))
        if len(temp) < 2:
            return False
    return True
#%%
def TracePolyline(mesh, start_node_idx, next_node_idx, end_node_idx=None, angle_threshold=np.pi/4):
    #find a smoothed polyline on mesh: start_node_idx -> next_node_idx -> ... -> end_node_idx
    #no self-interselction
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    #---------
    if mesh.node_to_node_adj_table is None:
        mesh.build_node_to_node_adj_table()
    node_adj_table=mesh.node_to_node_adj_table
    Polyline=[start_node_idx, next_node_idx]
    while True:
        idx_list_next=node_adj_table[Polyline[-1]]
        idx_list_next=list(set(idx_list_next)-set(Polyline))
        if len(idx_list_next) == 0:
            break
        if end_node_idx in idx_list_next:
            Polyline.append(end_node_idx)
            break
        angel_list=[]
        for k in range(0, len(idx_list_next)):
            idxA=Polyline[-2]
            idxB=Polyline[-1]
            idxC=idx_list_next[k]
            vector0=mesh.node[idxB]-mesh.node[idxA]
            vector1=mesh.node[idxC]-mesh.node[idxB]
            angle_k=ComputeAngleBetweenTwoVectorIn3D(vector0, vector1)
            angle_k=angle_k.view(-1).item()
            angel_list.append(angle_k)
        k_min=np.argmin(angel_list)
        if angel_list[k_min] > angle_threshold:
            break
        Polyline.append(idx_list_next[k_min])
    #done
    return Polyline
#%%
def MergeMesh(meshA, node_idx_listA, meshB, node_idx_listB, distance_threshold):
    #Merge meshA (larger) and meshB (smaller)
    #The shared points are in node_idx_listA of meshA and node_idx_listB of meshB
    #if the distance between two nodes is <= distance_threshold, then merge the two nodes
    if (not isinstance(meshA, Mesh)) or (not isinstance(meshA, Mesh)):
        raise NotImplementedError
    if meshA.node.shape[0] == 0:
        meshAB=Mesh(None, None, dtype=meshA.node.dtype, element_type=None, mesh_type=meshA.mesh_type)
        meshAB.copy(meshA.node, meshA.element)
    if meshB.node.shape[0] == 0:
        meshAB=Mesh(None, None, dtype=meshB.node.dtype, element_type=None, mesh_type=meshB.mesh_type)
        meshAB.copy(meshB.node, meshB.element)

    #node_idx_map_A_to_Out=np.arange(0, meshA.node.shape[0])
    node_idx_map_B_to_Out=-1*np.ones(meshB.node.shape[0])

    A = meshA.node[node_idx_listA]
    counterA = np.zeros(len(A))
    for n in range(0, len(node_idx_listB)):
        node_n=meshB.node[node_idx_listB[n]].view(1,-1)
        dist=((A-node_n)**2).sum(dim=1).sqrt()
        idx=dist.argmin()
        if dist[idx] <= distance_threshold:
            node_idx_map_B_to_Out[node_idx_listB[n]]=node_idx_listA[idx]
            counterA[idx]+=1
    if counterA.max() > 1:
        raise ValueError("two nodes of meshB are mapped to the same node of meshA")

    node_idx=meshA.node.shape[0]-1
    node_idx_listB_keep=[]
    for n in range(0, meshB.node.shape[0]):
        if node_idx_map_B_to_Out[n] < 0:
            node_idx+=1
            node_idx_map_B_to_Out[n]=node_idx
            node_idx_listB_keep.append(n)

    nodeAB=torch.cat([meshA.node, meshB.node[node_idx_listB_keep]], dim=0)
    elementAB=meshA.copy_element('list') + meshB.copy_element('list')
    for m in range(len(meshA.element), len(elementAB)):
        elm=elementAB[m]
        for k in range(0, len(elm)):
            elm[k]=node_idx_map_B_to_Out[elm[k]]

    meshAB=Mesh(nodeAB, elementAB, dtype=meshA.node.dtype, element_type=None, mesh_type=meshA.mesh_type)
    return meshAB