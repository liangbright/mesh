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
try:
    import vtk
except:
    print("cannot import vtk")
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
def ExtractRegionEnclosedByCurve(mesh, node_curve_list, inner_element_idx):
    #node_curve_list[k] is a curve - represented by a list/array of node indexes on mesh
    #the combined curve (from curve_list[0] to curve_list[-1]) is closed
    #inner_element_idx is the index of an element inside the region
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #-------------------------
    curve=[]
    for k in range(0, len(node_curve_list)):
        curve_k=[int(x) for x in node_curve_list[k]]
        curve.extend(curve_k)
    flag_close, idx_bad=IsCurveClosed(mesh, curve)
    if not flag_close:
        #raise ValueError('curve is not closed at node '+str(idx_bad))
        print('curve may be open or self-intersect at node '+str(idx_bad)+" @ ExtractRegionEnclosedByCurve")
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
    region_element_list=[inner_element_idx]
    active_element_list=[inner_element_idx]
    counter=0
    element_flag=np.zeros(len(mesh.element))#flag 1: in the  region; 0: not in the region
    element_flag[inner_element_idx]=1
    while True:
        new_active_element_list=[]
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
                    if len(adj_elm_idx) > 2:
                        raise ValueError("len(adj_elm_idx) > 2")
                    elif len(adj_elm_idx) == 2:
                        if adj_elm_idx[0] == act_elm_idx:
                            new_active_element_list.append(adj_elm_idx[1])
                        else:
                            new_active_element_list.append(adj_elm_idx[0])
        if len(new_active_element_list) == 0:
            break
        new_active_element_list=np.unique(new_active_element_list)
        active_element_list=new_active_element_list[element_flag[new_active_element_list]==0].tolist()
        element_flag[active_element_list]=1
        #active_element_list=list(set(new_active_element_list)-set(region_element_list))
        region_element_list.extend(active_element_list)
        counter+=1
        #if counter ==100:
        #    break
    #--------------
    return region_element_list
#%%
def SimpleSmootherForMeshNodeNormal(mesh, lamda, mask, n_iters, update_node_normal=True):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if update_node_normal == True:
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
        merged_mesh=MergeMesh(merged_mesh, merged_mesh.find_boundary_node(),
                              mesh_n, mesh_n.find_boundary_node(),
                              distance_threshold)
    merged_mesh=PolygonMesh(merged_mesh.node, merged_mesh.element)
    return merged_mesh
#%%
def SimpleSmootherForQuadMesh(mesh, lamda, mask, n_iters):
    if mesh.node_to_element_adj_table is None:
        mesh.build_node_to_element_adj_table()
    try:
        adj_link=mesh.mesh_data['quad_node_to_node_adj_link']
    except:
        adj_link=[]
        for n in range(0, len(mesh.node)):
            adj_elm_idx=mesh.node_to_element_adj_table[n]
            adj_elm=mesh.element[adj_elm_idx]
            adj_node_idx=torch.unique(adj_elm.reshape(-1)).tolist()
            for idx in adj_node_idx:
                if idx != n:
                    adj_link.append([idx, n])
                    adj_link.append([n, idx])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        mesh.mesh_data['quad_node_to_node_adj_link']=adj_link
    for n in range(0, n_iters):
        SimpleSmoother(mesh.node, adj_link, lamda, mask, inplace=True)
#%%
def ProjectPointToSurface(mesh, point, mesh_vtk=None):
    #ProjectPointToFaceByVTKCellLocator in MDK
    #point (N, 3)
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if mesh.is_tri() == False:
        raise NotImplementedError
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    CellLocator = vtk.vtkCellLocator()
    CellLocator.SetDataSet(mesh_vtk)
    CellLocator.BuildLocator()
    point_proj=[]
    face_proj=[]
    for k in range(0, len(point)):
        testPoint = [float(point[k][0]), float(point[k][1]), float(point[k][2])]
        closestPoint=[0, 0, 0] #the coordinates of the closest point will be returned here
        closestPointDist2=vtk.reference(0) #the squared distance to the closest point will be returned here
        cellId= vtk.reference(0); #the cell id of the cell containing the closest point will be returned here
        subId= vtk.reference(0); #this is rarely used (in triangle strips only, I believe)
        CellLocator.FindClosestPoint(testPoint, closestPoint, cellId, subId, closestPointDist2);
        point_proj.append(closestPoint)
        face_proj.append(int(cellId))
    return point_proj, face_proj
#%%
def SmoothAndProject(mesh_move, mesh_fixed, lamda, mask, n1_iters, n2_iters):
    #smooth mesh_move and project it to mesh_fixed
    mesh_fixed_vtk=mesh_fixed.convert_to_vtk()
    for k in range(0, n2_iters):
        SimpleSmootherForMesh(mesh_move, lamda, mask, n1_iters)
        node_proj, face_proj=ProjectPointToSurface(mesh_fixed, mesh_move.node, mesh_fixed_vtk)
        node_proj=torch.tensor(node_proj, dtype=mesh_move.node.dtype)
        temp=mask.view(-1)
        mesh_move.node[temp>0]=node_proj[temp>0]





