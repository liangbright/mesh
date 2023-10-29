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
def FindConnectedRegion(mesh, ref_element_idx, adj='edge'):
    if not isinstance(mesh, PolygonMesh):
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
def SimpleSmoother(field, adj_link, lamda, mask, inplace):
    #field.shape (N, ?)
    if mask is None:
        mask=1
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
def SimpleMeshSmoother(mesh, lamda, mask, n_iters):
    #lamda: x_i = x_i + lamda*mean_j(x_j - x_i),  0<=lamda<=1
    #inplace: True or False
    #if mask is not None: mask[k]: 1 to smooth the node-k; 0 not to smooth the node-k
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link
    for n in range(0, n_iters):
        SimpleSmoother(mesh.node, adj_link, lamda, mask, inplace=True)
#%%
def SmoothNodeNormal(mesh, lamda, mask, n_iters, angle_weighted=False):
    if not isinstance(mesh, TriangleMesh):
        raise NotImplementedError
    mesh.update_node_normal(angle_weighted=angle_weighted)
    if mesh.node_to_node_adj_link is None:
        mesh.build_node_to_node_adj_link()
    adj_link=mesh.node_to_node_adj_link
    node_normal=mesh.node_normal
    for n in range(0, n_iters):
        SimpleSmoother(node_normal, adj_link, lamda, mask, inplace=True)
        normal_norm=torch.norm(node_normal, p=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        node_normal=node_normal/normal_norm
    node_normal=node_normal.contiguous()
    mesh.node_normal=node_normal
#%%
def TracePolyline(mesh, start_node_idx, next_node_idx, end_node_idx=None, angle_threshold=np.pi/4):
    #find a smoothed polyline on mesh: start_node_idx -> next_node_idx -> ... -> end_node_idx
    #no self-interselction
    if not isinstance(mesh, PolygonMesh):
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
    if (not isinstance(meshA, PolygonMesh)) or (not isinstance(meshA, PolygonMesh)):
        raise NotImplementedError
    if meshA.node.shape[0] == 0:
        meshAB=PolygonMesh()
        meshAB.copy(meshA.node, meshA.element)
    if meshB.node.shape[0] == 0:
        meshAB=PolygonMesh()
        meshAB.copy(meshB.node, meshB.element)

    #node_idx_map_A_to_Out=np.arange(0, meshA.node.shape[0])
    node_idx_map_B_to_Out=-1*np.ones(meshB.node.shape[0])

    curveA = meshA.node[node_idx_listA]
    counterA = np.zeros(len(curveA))
    for n in range(0, len(node_idx_listB)):
        node_n=meshB.node[node_idx_listB[n]].view(1,-1)
        dist=((curveA-node_n)**2).sum(dim=1).sqrt()
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

    meshAB=PolygonMesh(nodeAB, elementAB)
    return meshAB
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
