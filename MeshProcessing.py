# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:12:58 2023

@author: liang
"""
import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from Mesh import Mesh
from copy import deepcopy
#%%
def ComputeAngleBetweenTwoVectorIn3D_slow(VectorA, VectorB):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.view(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.view(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        eps1=1e-12
        eps2=1e-7 # torch.acos grad issue, it must be 1e-7        
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        eps1=1e-12
        eps2=1e-16
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = np.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if np.any(L2Norm_A <= eps1) or np.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, np.clip to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        L2Norm_A=np.clip(L2Norm_A, min=eps1)
        L2Norm_B=np.clip(L2Norm_B, min=eps1)
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, min=-1, max=1)
        Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA,  torch.Tensor):
        L2Norm_A = torch.sqrt(VectorA[:,0]*VectorA[:,0]+VectorA[:,1]*VectorA[:,1]+VectorA[:,2]*VectorA[:,2])
        L2Norm_B = torch.sqrt(VectorB[:,0]*VectorB[:,0]+VectorB[:,1]*VectorB[:,1]+VectorB[:,2]*VectorB[:,2])
        if torch.any(L2Norm_A <= eps1) or torch.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, torch.clamp to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        with torch.no_grad():
            L2Norm_A.data.clamp_(min=eps1)
            L2Norm_B.data.clamp_(min=eps1)                
        CosTheta = (VectorA[:,0]*VectorB[:,0]+VectorA[:,1]*VectorB[:,1]+VectorA[:,2]*VectorB[:,2])/(L2Norm_A*L2Norm_B);              
        CosTheta = torch.clamp(CosTheta, min=-1+eps2, max=1-eps2)
        Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
    return Theta
    '''
    #https://github.com/pytorch/pytorch/issues/8069  
    eps=1e-16
    x=torch.tensor(-1.0, requires_grad=True, dtype=torch.float64)
    x1=torch.clamp(x, min=-1+eps, max=1-eps)
    Theta = torch.acos(x1)
    Theta.backward()
    print(x.grad)
    '''
#%%
def ComputeAngleBetweenTwoVectorIn3D(VectorA, VectorB, return_cos=False):
    #angle from A to B, right hand rule
    #angle ~[0 ~2pi]
    #VectorA.shape (B,3)
    #VectorB.shape (B,3)
    if len(VectorA.shape) == 1:
        VectorA=VectorA.reshape(1,3)
    if len(VectorB.shape) == 1:
        VectorB=VectorB.reshape(1,3)
    
    if VectorA.dtype != VectorB.dtype:
        raise ValueError
    
    if VectorA.dtype == np.float32 or VectorA.dtype == torch.float32:
        eps1=1e-12
        eps2=1e-7 # torch.acos grad issue, it must be 1e-7        
    elif VectorA.dtype == np.float64 or VectorA.dtype == torch.float64:
        eps1=1e-12
        eps2=1e-16
    
    if isinstance(VectorA, np.ndarray) and isinstance(VectorA, np.ndarray):
        L2Norm_A = np.linalg.norm(VectorA, ord=2, axis=-1)
        L2Norm_B = np.linalg.norm(VectorB, ord=2, axis=-1)
        if np.any(L2Norm_A <= eps1) or np.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, np.clip to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        L2Norm_A=np.clip(L2Norm_A, a_min=eps1, a_max=np.inf)
        L2Norm_B=np.clip(L2Norm_B, a_min=eps1, a_max=np.inf)
        CosTheta = np.sum(VectorA*VectorB, axis=-1)/(L2Norm_A*L2Norm_B);
        CosTheta = np.clip(CosTheta, a_min=-1, a_max=1)
        if return_cos == False:
            Theta = np.arccos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
    elif isinstance(VectorA, torch.Tensor) and isinstance(VectorA, torch.Tensor):
        L2Norm_A=norm(VectorA, ord=2, dim=-1)
        L2Norm_B=norm(VectorB, ord=2, dim=-1)        
        if torch.any(L2Norm_A <= eps1) or torch.any(L2Norm_B <= eps1):
            print("L2Norm <= eps, torch.clamp to eps @ ComputeAngleBetweenTwoVectorIn3D(...)")
        with torch.no_grad():
            L2Norm_A.data.clamp_(min=eps1)
            L2Norm_B.data.clamp_(min=eps1)         
        CosTheta = (VectorA*VectorB).sum(dim=-1)/(L2Norm_A*L2Norm_B);                      
        if return_cos == False:
            CosTheta = torch.clamp(CosTheta, min=-1+eps2, max=1-eps2)
            Theta = torch.acos(CosTheta) #[0, pi], acos(-1) = pi
            return Theta    
        else:
            return CosTheta
#%%
def FindConnectedRegion(mesh, ref_element_idx, adj):
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    if adj not in ["node", "edge", "face"]:
        raise ValueError()
    if mesh.element_to_element_adj_table[adj] is None:
        mesh.build_element_to_element_adj_table(adj=adj)
    element_adj_table=mesh.element_to_element_adj_table[adj]
    region_element_list=[ref_element_idx]
    active_element_list=[ref_element_idx]
    while True:
        new_active_element_list=[]
        region_element_set=set(region_element_list)
        for elm_idx in active_element_list:
            adj_elm_idxlist=element_adj_table[elm_idx]
            adj_elm_idxlist=list(set(adj_elm_idxlist)-region_element_set)
            new_active_element_list.extend(adj_elm_idxlist)
        if len(new_active_element_list) ==0:
            break
        active_element_list=np.unique(new_active_element_list).tolist()
        region_element_list.extend(active_element_list)
    #indexes of elements
    return region_element_list
#%%
def SegmentMeshToConnectedRegion(mesh, adj):
    element_list=np.arange(0, len(mesh.element)).tolist()
    region_list=[]
    while True:
        if len(element_list) == 0:
            break
        region=FindConnectedRegion(mesh, element_list[0], adj)
        region_list.append(region)
        element_list=list(set(element_list)-set(region))
    return region_list
#%%
def SimpleSmoother(field, adj_link, lamda, mask, inplace):
    #field.shape (N, ?)
    if mask is None:
        mask=1    
    elif isinstance(mask, list):
        mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, np.ndarray):
        mask=torch.tensor(mask, dtype=field.dtype, device=field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, torch.Tensor):
        mask=mask.to(field.dtype).to(field.device)
        mask=mask.reshape(-1,1)
    elif isinstance(mask, int) or isinstance(mask, float):
        if mask != 1:
            raise ValueError            
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
    #mesh.node is modified
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
            return False, idx
    return True, None
#%%
def TracePolyline(mesh, start_node_idx, next_node_idx, end_node_idx=None, angle_threshold=np.pi/2):
    #find a smoothed polyline on mesh: start_node_idx -> next_node_idx -> ... -> end_node_idx
    #no self-interselction
    if not isinstance(mesh, Mesh):
        raise NotImplementedError
    #---------
    start_node_idx=int(start_node_idx)
    next_node_idx=int(next_node_idx)
    if end_node_idx is not None:
        end_node_idx=int(end_node_idx)
        if end_node_idx == start_node_idx:
            raise ValueError("start_node_idx is end_node_idx")
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
        else:
            Polyline.append(idx_list_next[k_min])
    if (end_node_idx is not None) and (Polyline[-1] != end_node_idx):
        print("warning: Polyline[-1] != end_node_idx @ TracePolyline")
    #done
    return Polyline
#%%
def MergeMesh(meshA, node_idx_listA, meshB, node_idx_listB, distance_threshold):
    #Merge: meshA <= meshB
    #The shared points are in node_idx_listA of meshA and node_idx_listB of meshB
    #if both node_idx_listA and node_idx_listB are None, simply do meshA + meshB - no point merging    
    #if the distance between two nodes is <= distance_threshold, then merge the two nodes
    if (not isinstance(meshA, Mesh)) or (not isinstance(meshA, Mesh)):
        #raise NotImplementedError
        print("warning: input may not be Mesh @ MergeMesh")
    if meshA.node.shape[0] == 0 or meshB.node.shape[0] == 0:
        raise ValueError("meshA or meshB is empty")
    #---------------------------------------------------
    if (node_idx_listA is None) or (node_idx_listB is None):
        elementA=meshA.make_element_copy('list')
        if not isinstance(meshB.element, list):
            elementB=deepcopy(meshB.element)+len(meshA.node)
            elementB=elementB.tolist()
        else:
            elementB=deepcopy(meshB.element)
            for m in range(0, len(elementB)):
                for n in range(0, len(elementB[m])):
                    elementB[m][n]=elementB[m][n]+len(meshA.node)
        nodeAB=meshA.node.tolist() + meshB.node.tolist()
        elementAB=elementA + elementB
        meshAB=Mesh(nodeAB, elementAB, meshA.element_type, meshA.mesh_type)
        return meshAB
    #---------------------------------------------------
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
    elementAB=meshA.make_element_copy('list') + meshB.make_element_copy('list')
    for m in range(len(meshA.element), len(elementAB)):
        elm=elementAB[m]
        for k in range(0, len(elm)):
            elm[k]=node_idx_map_B_to_Out[elm[k]]

    meshAB=Mesh(nodeAB, elementAB, meshA.element_type, meshA.mesh_type)
    return meshAB
#%%
def RemoveUnusedNode(mesh, return_node_idx_list=False, clear_adj_info=True):
    #if a node does not belong to an element, then it is unused
    #this function may change the node order in mesh.node
    element=mesh.copy_element("list")
    map=-torch.ones(mesh.node.shape[0], dtype=torch.int64)
    node_idx=-1
    for m in range(0, len(element)):
        elm=element[m]
        for n in range(0, len(elm)):
            idx=elm[n]
            if idx < 0:
                raise ValueError("idx < 0, the output may be wrong")
            if map[idx] < 0:
                node_idx+=1
                elm[n]=node_idx#new idx
                map[idx]=node_idx
            else:
                elm[n]=map[idx]
    node_idx_list=torch.where(map>=0)[0]
    node=mesh.node[node_idx_list]
    mesh.copy(node, element)
    if clear_adj_info == True:
        mesh.clear_adj_info()
    if return_node_idx_list == True:
        return node_idx_list
#%%
def FindNearestNode(mesh, point, distance_threshold=np.inf):
    #point (K, 3) or (3,)
    if not isinstance(point, (torch.Tensor, np.ndarray)):
        raise ValueError("unsupported type "+str(type(point)))
    if isinstance(point, np.ndarray):
        point=torch.tensor(point, dtype=mesh.node.dtype)    
    flag=(len(point.shape) == 1)
    point=point.view(-1,3)
    node_idx_list=[]
    for n in range(0, len(point)):
        p=point[n].view(1,-1)
        d=((mesh.node-p)**2).sum(dim=-1)
        idx=d.argmin()
        if float(d[idx]) <= distance_threshold:
            node_idx_list.append(int(idx))    
    if flag == False:
        return node_idx_list
    else:
        return node_idx_list[0]
#%%
def FindNeighborNode(mesh, node_idx, max_n_hop):
    if isinstance(node_idx, int):
        node_idx=[node_idx]
    elif isinstance(node_idx, tuple):
        node_idx=list(node_idx)
    elif isinstance(node_idx, list):
        pass
    else:
        raise ValueError('node_idx must be int, tuple, or list')
    mesh.build_node_to_node_adj_table()
    adj_table=mesh.node_to_node_adj_table
    neighbor_node_list=node_idx
    active_node_list=node_idx
    n_hop=0
    while True:
        new_active_node_list=[]
        node_set=set(neighbor_node_list)
        for idx in active_node_list:
            adj_idx_list=adj_table[idx]
            adj_idx_list=list(set(adj_idx_list)-node_set)
            new_active_node_list.extend(adj_idx_list)
        if len(new_active_node_list) ==0:
            break
        active_node_list=np.unique(new_active_node_list).tolist()
        neighbor_node_list.extend(active_node_list)
        n_hop+=1
        if n_hop >= max_n_hop:
            break
    return neighbor_node_list
    
        
        
        
        
    
    
    
    
