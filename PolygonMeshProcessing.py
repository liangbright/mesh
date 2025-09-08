import torch
from torch.linalg import vector_norm as norm
import torch_scatter
import numpy as np
from PolylineMesh import PolylineMesh
from PolygonMesh import PolygonMesh
from TriangleMesh import TriangleMesh
from Tri6Mesh import Tri6Mesh
from QuadMesh import QuadMesh
from QuadTriangleMesh import QuadTriangleMesh
from copy import deepcopy
from MeshProcessing import (SimpleSmoother, SimpleSmootherForMesh,
                            ComputeAngleBetweenTwoVectorIn3D, TracePolyline,
                            IsCurveClosed, MergeMesh,
                            FindConnectedRegion, FindNearestNode, SegmentMeshToConnectedRegion,
                            FindNeighborNode)
try:
    import vtk
except:
    print("cannot import vtk")
#%%
def TraceMeshBoundaryCurve(mesh, start_node_idx, next_node_idx=None, end_node_idx=None):
    #trace boundary starting from start_node_idx -> next_node_idx -> ...
    #this function may not work well if two boundary curves share points
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    #---------
    start_node_idx=int(start_node_idx)
    if next_node_idx is not None:
        next_node_idx=int(next_node_idx)
    if end_node_idx is not None:
        end_node_idx=int(end_node_idx)
        if end_node_idx == start_node_idx:
            raise ValueError("start_node_idx is end_node_idx")
    #---------
    if mesh.node_to_edge_adj_table is None:
        mesh.build_node_to_edge_adj_table()
    #---------
    boundary_node, boundary_edge=mesh.find_boundary_node_and_edge()
    BoundaryCurve=[]
    if start_node_idx not in boundary_node:
        raise ValueError('start_node('+str(start_node_idx)+') is not a boundary node')
    if next_node_idx is not None:
        if next_node_idx not in boundary_node:
            raise ValueError('next_node('+str(next_node_idx)+') is not a boundary node')
        edge_idx=mesh.get_edge_idx_from_node_pair(start_node_idx, next_node_idx) 
        if edge_idx not in boundary_edge:
            raise ValueError('no boundary edge between node('+str(start_node_idx)+') and node('+str(next_node_idx)+')')
        else:
            BoundaryCurve.append(start_node_idx)
            start_node_idx=next_node_idx
    if end_node_idx is not None:
        if end_node_idx not in boundary_node:
            raise ValueError('end_node('+str(end_node_idx)+') is not a boundary node')
            #return BoundaryCurve
    #---------
    idx_next=start_node_idx
    while True:
        BoundaryCurve.append(idx_next)
        if idx_next == end_node_idx:
            break
        edge_idx_list=mesh.node_to_edge_adj_table[idx_next]
        flag=False
        for edge_idx in edge_idx_list:
            if edge_idx in boundary_edge:
                idx_list=mesh.edge[edge_idx].tolist()
                if idx_next == idx_list[0]:
                    node_idx=idx_list[1]
                elif idx_next == idx_list[1]:
                    node_idx=idx_list[0]
                else:
                    raise ValueError('something is wrong in edge and node_to_edge_adj_table')
                if node_idx not in BoundaryCurve:
                    idx_next=node_idx
                    flag=True
                    break
        if flag == False:
            break
    return BoundaryCurve
#%%
def FindMeshBoundaryCurve(mesh):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    node_idx_list=mesh.find_boundary_node()
    BoundaryCurveList=[]
    while True:
        if len(node_idx_list) == 0:
            break
        BoundaryCurve=TraceMeshBoundaryCurve(mesh, node_idx_list[0])
        BoundaryCurveList.append(BoundaryCurve)
        node_idx_list=list(set(node_idx_list)-set(BoundaryCurve))
    return BoundaryCurveList
#%%
def ExtractRegionEnclosedByCurve(mesh, node_curve_list, inner_element_idx, max_n_elements=float('inf')):
    #node_curve_list[k] is a curve - represented by a list/array of node indexes on mesh
    #the combined curve (from curve_list[0] to curve_list[-1]) is closed
    #inner_element_idx is the index of an element inside the region
    #max_n_elements: maximum number of elements in the region
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
    if mesh.element_to_edge_adj_table is None:
        mesh.build_element_to_edge_adj_table()
    element_to_edge_adj_table=mesh.element_to_edge_adj_table    
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
            edge_idx_list=element_to_edge_adj_table[act_elm_idx]
            for edge_idx in edge_idx_list:
                if edge_idx not in edge_curve:
                    adj_elm_idx=edge_to_element_adj_table[edge_idx]
                    if len(adj_elm_idx) > 2:
                        raise ValueError("len(adj_elm_idx) > 2 at edge "+str(edge_idx))
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
        if len(region_element_list) > max_n_elements:
            print('break: len(region_element_list) > max_n_elements @ExtractRegionEnclosedByCurve')
            break
    #--------------
    # indexes of elements in the region
    return region_element_list  
#%%
def SegmentMeshByCurve(mesh, node_curve_list):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    element_list=np.arange(0, len(mesh.element)).tolist()
    region_list=[]
    while True:
        if len(element_list) == 0:
            break
        region=ExtractRegionEnclosedByCurve(mesh, node_curve_list, element_list[0])
        region_list.append(region)
        element_list=list(set(element_list)-set(region))
    return region_list
#%%
def MergeMeshOnBoundary(mesh_list, distance_threshold):
    merged_mesh=mesh_list[0]
    for n in range(1, len(mesh_list)):
        mesh_n=mesh_list[n]
        if not isinstance(mesh_n, PolygonMesh):
            raise NotImplementedError
        merged_mesh=MergeMesh(merged_mesh, merged_mesh.find_boundary_node(),
                              mesh_n, mesh_n.find_boundary_node(),
                              distance_threshold)
        merged_mesh=PolygonMesh(merged_mesh.node, merged_mesh.element)
    return merged_mesh
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
        normal_norm=norm(node_normal, ord=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        node_normal=node_normal/normal_norm
    node_normal=node_normal.contiguous()
    mesh.node_normal=node_normal
#%%
def SimpleSmootherForMeshElementNormal(mesh, lamda, mask, n_iters, update_element_normal=True):
    if not isinstance(mesh, PolygonMesh):
        raise NotImplementedError
    if update_element_normal == True:
        mesh.update_element_area_and_normal()
    element_normal =mesh.element_normal 
    if mesh.element_to_element_adj_link['node'] is None:
        mesh.build_element_to_element_adj_link('node')
    adj_link=mesh.element_to_element_adj_link['node']
    for n in range(0, n_iters):
        SimpleSmoother(element_normal, adj_link, lamda, mask, inplace=True)
        normal_norm=norm(element_normal, ord=2, dim=1, keepdim=True)
        normal_norm=normal_norm.clamp(min=1e-12)
        element_normal=element_normal/normal_norm
    element_normal=element_normal.contiguous()
    mesh.element_normal=element_normal    
#%%
def SimpleSmootherForQuadMesh(mesh, lamda, mask, n_iters):
    if not isinstance(mesh, QuadMesh):
        raise NotImplementedError
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
#dtype is a necessary parameter of a function that is based on vtk functions
# function(mesh, mesh_vtk, dtype) where mesh_vtk is the output of some vtk function and mesh does not exist
#%%
def CutMeshByCurve(mesh, curve, point_ref, straight_cut=False, return_unselected=False, 
                   clean_output=False, eps=1e-5, triangulate_output=False, mesh_vtk=None, dtype=None,
                   threshold=0):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    curve_vtk=vtk.vtkPoints()
    curve_vtk.SetDataTypeToDouble()
    curve_vtk.SetNumberOfPoints(len(curve))
    for n in range(0, len(curve)):
        curve_vtk.SetPoint(n, float(curve[n][0]), float(curve[n][1]), float(curve[n][2]))
    selecter=vtk.vtkSelectPolyData()
    selecter.SetInputData(mesh_vtk)
    selecter.SetLoop(curve_vtk)
    selecter.SetEdgeSearchModeToDijkstra()
    selecter.SetSelectionModeToClosestPointRegion()
    selecter.SetClosestPoint(float(point_ref[0]), float(point_ref[1]), float(point_ref[2]))    
    selecter.SetGenerateSelectionScalars(straight_cut)
    selecter.SetGenerateUnselectedOutput(return_unselected)
    selecter.Update()
    if straight_cut == False:
        output_vtk=selecter.GetOutput()
        if return_unselected == True:
            unselected_output_vtk=selecter.GetUnselectedOutput()
    else:
        clipper=vtk.vtkClipPolyData()
        clipper.SetInputData(selecter.GetOutput())
        clipper.SetValue(threshold)
        clipper.SetInsideOut(True)
        if return_unselected == True:
            clipper.GenerateClippedOutputOn()
        clipper.Update()
        #vtkClipPolyData may have weird output: isoloated points not in the selected region
        #use vtkConnectivityFilter to remove those isoloated points
        extractor=vtk.vtkConnectivityFilter() 
        extractor.SetInputData(clipper.GetOutput())
        extractor.SetExtractionModeToAllRegions()
        extractor.Update()
        output_vtk=extractor.GetOutput()        
        if return_unselected == True:            
            extractor=vtk.vtkConnectivityFilter()
            extractor.SetInputData(clipper.GetClippedOutput())
            extractor.SetExtractionModeToAllRegions()
            extractor.Update()
            unselected_output_vtk=extractor.GetOutput()
    #----------------------------------------
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=ConvertPolygonMeshToTriangleMesh(None, output_vtk, dtype)
    if return_unselected == False:
        return output_mesh
    #----------------------------------------
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(unselected_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        unselected_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        unselected_output_mesh=PolygonMesh()
        unselected_output_mesh.read_mesh_vtk(unselected_output_vtk, dtype)
    else:
        unselected_output_mesh=ConvertPolygonMeshToTriangleMesh(None, unselected_output_vtk, dtype)
    return output_mesh, unselected_output_mesh
#%%
def ProjectPointToMesh(mesh, point, mesh_vtk=None, dtype=None):
    #point (N, 3)
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    #------------------------------------------
    if torch.is_tensor(point) or isinstance(point, np.ndarray()):
        if len(point.shape) == 1:
            point=point.reshape(1,3)
        if len(point.shape) != 2:
            raise ValueError('point.shape is not supported')
        if point.shape[1] != 3:
            raise ValueError('point.shape is not supported')
    else:
        raise ValueError('point can only be torch.tensor or np.ndarray')
    #------------------------------------------
    CellLocator=vtk.vtkStaticCellLocator()
    CellLocator.SetDataSet(mesh_vtk)
    CellLocator.BuildLocator()
    point_proj=[]
    element_proj=[]
    for k in range(0, len(point)):
        testPoint=[float(point[k][0]), float(point[k][1]), float(point[k][2])]
        closestPoint=[0, 0, 0] #the coordinates of the closest point will be returned here
        closestPointDist2=vtk.reference(0) #the squared distance to the closest point will be returned here
        cellId=vtk.reference(0); #the cell id of the cell containing the closest point will be returned here
        subId=vtk.reference(0);  #this is rarely used (in triangle strips only, I believe)
        CellLocator.FindClosestPoint(testPoint, closestPoint, cellId, subId, closestPointDist2);
        point_proj.append(closestPoint)
        element_proj.append(int(cellId))
    point_proj=torch.tensor(point_proj, dtype=dtype)
    return point_proj, element_proj
#%%
def SmoothAndProject(mesh_move, mesh_fixed, lamda, mask, n1_iters, n2_iters, mesh_fixed_vtk=None, smooth_first=True):
    #smooth mesh_move and project it to mesh_fixed
    #mesh_move.node is modified
    #mesh_fixed must be a triangle mesh
    if mesh_fixed_vtk is None:
        mesh_fixed_vtk=mesh_fixed.convert_to_vtk()
    for k in range(0, n2_iters):
        if smooth_first == True:
            SimpleSmootherForMesh(mesh_move, lamda, mask, n1_iters)
        node_proj, element_proj=ProjectPointToMesh(mesh_fixed, mesh_move.node, mesh_fixed_vtk)        
        temp=mask.view(-1)
        mesh_move.node[temp>0]=node_proj[temp>0]
        if smooth_first == False and k < n2_iters-1:
            SimpleSmootherForMesh(mesh_move, lamda, mask, n1_iters)
#%%
def ChamferDistance(meshA, meshB, reduction):
    #this function is not differentiable
    #perhaps, we should not name it ChamferDistance
    nodeB_proj, __=ProjectPointToMesh(meshA, meshB.node)
    nodeA_proj, __=ProjectPointToMesh(meshB, meshA.node)
    distA=((meshA.node-nodeA_proj)**2).sum(dim=1).sqrt()
    distB=((meshB.node-nodeB_proj)**2).sum(dim=1).sqrt()
    if reduction == 'none':
        return distA, distB
    elif reduction == 'mean':
        return 0.5*(distA.mean()+distB.mean())
    else:
        raise ValueError
#%%
def ConvertPolygonMeshToTriangleMesh(mesh, mesh_vtk=None, dtype=None):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    trifilter=vtk.vtkTriangleFilter()
    trifilter.SetInputData(mesh_vtk)
    trifilter.Update()
    output_mesh=TriangleMesh()
    output_mesh.read_mesh_vtk(trifilter.GetOutput(), dtype)
    return output_mesh
#%%
def ClipMeshByPlane(mesh, origin, normal, return_clipped_output=False, clean_output=False, eps=1e-5,
                    triangulate_output=False, mesh_vtk=None, dtype=None):
    #origin and normal define the cut plane
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    plane=vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    clipper=vtk.vtkClipPolyData()
    clipper.SetInputData(mesh_vtk)
    clipper.SetClipFunction(plane)
    clipper.SetValue(0)
    if return_clipped_output == True:
        clipper.GenerateClippedOutputOn()
    clipper.Update()
    output_vtk=clipper.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=ConvertPolygonMeshToTriangleMesh(None, output_vtk, dtype)
    if return_clipped_output == False:
        return output_mesh
    #----------------------------------------
    clipped_output_vtk=clipper.GetClippedOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(clipped_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        clipped_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        clipped_output_mesh=PolygonMesh()
        clipped_output_mesh.read_mesh_vtk(clipped_output_vtk, dtype)
    else:
        clipped_output_mesh=ConvertPolygonMeshToTriangleMesh(None, clipped_output_vtk, dtype)
    return output_mesh, clipped_output_mesh
#%%
def ClipMeshByAttribute(mesh, attribute_threshold, node_attribute_name=None, element_attribute_name=None,
                        invert=True, return_clipped_output=False, clean_output=False, eps=1e-5,
                        triangulate_output=False, mesh_vtk=None, dtype=None):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    if (node_attribute_name is not None) and (element_attribute_name is None):
        attribute_name=node_attribute_name
        if attribute_name not in mesh.node_data.keys():
            raise ValueError('wrong node_attribute_name')
        mesh_vtk.GetPointData().SetActiveScalars(attribute_name)        
    elif (node_attribute_name is None) and (element_attribute_name is not None):
        attribute_name=element_attribute_name        
        if attribute_name not in mesh.element_data.keys():
            raise ValueError('wrong element_attribute_name')
        mesh_vtk.GetCellData().SetActiveScalars(attribute_name)        
    elif (node_attribute_name is not None) and (element_attribute_name is not None):
        raise ValueError('use node_attribute_name or element_attribute_name, not both')
    else:
        raise ValueError('invalid attribute_name')
    clipper=vtk.vtkClipPolyData()
    clipper.SetInputData(mesh_vtk)
    clipper.SetValue(attribute_threshold)
    clipper.SetInsideOut(invert)
    if return_clipped_output == True:
        clipper.GenerateClippedOutputOn()
    clipper.Update()
    output_vtk=clipper.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=ConvertPolygonMeshToTriangleMesh(None, output_vtk, dtype)
    if return_clipped_output == False:
        return output_mesh
    #----------------------------------------
    clipped_output_vtk=clipper.GetClippedOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(clipped_output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        clipped_output_vtk=cleaner.GetOutput()
    if triangulate_output == False:
        clipped_output_mesh=PolygonMesh()
        clipped_output_mesh.read_mesh_vtk(clipped_output_vtk, dtype)
    else:
        clipped_output_mesh=ConvertPolygonMeshToTriangleMesh(None, clipped_output_vtk, dtype)
    return output_mesh, clipped_output_mesh
#%%
def SliceMeshByPlane(mesh, origin, normal, clean_output=False, eps=1e-5, mesh_vtk=None, dtype=None):
    #similar to slice function in paraview
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    plane=vtk.vtkPlane()
    plane.SetOrigin(float(origin[0]), float(origin[1]), float(origin[2]))
    plane.SetNormal(float(normal[0]), float(normal[1]), float(normal[2]))
    if mesh is not None:
        if mesh.is_tri() == True:
            slicer=vtk.vtkPolyDataPlaneCutter()
        else:
            slicer=vtk.vtkPlaneCutter()
    else:
        slicer=vtk.vtkPlaneCutter()
    slicer.SetInputData(mesh_vtk)
    slicer.SetPlane(plane)
    slicer.Update()
    output_vtk=slicer.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    output_mesh=PolylineMesh()
    output_mesh.read_mesh_vtk(output_vtk, dtype=dtype)
    return output_mesh
#%%
def ComputeCurvature(mesh, curvature_name='mean', mesh_vtk=None, dtype=None):
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    cc = vtk.vtkCurvatures()
    cc.SetInputData(mesh_vtk)
    if 'gaussian' in curvature_name.lower():
        curvature_name='Gaussian_Curvature'
        cc.SetCurvatureTypeToGaussian()
        cc.Update()
    elif 'mean' in curvature_name.lower():
        curvature_name='Mean_Curvature'
        cc.SetCurvatureTypeToMean()
        cc.Update()
    elif 'max' in curvature_name.lower():
        curvature_name='Max_Curvature'
        cc.SetCurvatureTypeToMaximum()
        cc.Update()
    elif 'min' in curvature_name.lower():
        curvature_name='Min_Curvature'
        cc.SetCurvatureTypeToMinimum()
        cc.Update()
    else:
        raise ValueError('uknown curvature_name'+curvature_name)
    data=cc.GetOutput().GetPointData().GetAbstractArray(curvature_name)
    curvature=torch.zeros((data.GetNumberOfTuples(), ), dtype=dtype)
    for i in range(0, curvature.shape[0]):
        curvature[i]=data.GetComponent(i,0)
    return curvature
#%%
def FillHole(mesh, hole_size, clean_output=False, eps=1e-5, triangulate_output=False, mesh_vtk=None, dtype=None):
    #make a watertight mesh
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    filter=vtk.vtkFillHolesFilter()
    filter.SetInputData(mesh_vtk)
    filter.SetHoleSize(hole_size)
    filter.Update()
    output_vtk=filter.GetOutput()
    if clean_output == True:
        cleaner=vtk.vtkStaticCleanPolyData()
        cleaner.SetInputData(output_vtk)
        cleaner.ToleranceIsAbsoluteOn()
        cleaner.SetAbsoluteTolerance(eps)
        cleaner.RemoveUnusedPointsOn()
        cleaner.Update()
        output_vtk=cleaner.GetOutput()
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(output_vtk)
    normals.ConsistencyOn()
    normals.SplittingOff()
    normals.Update()    
    output_vtk=normals.GetOutput()
    if triangulate_output == False:
        output_mesh=PolygonMesh()
        output_mesh.read_mesh_vtk(output_vtk, dtype)
    else:
        output_mesh=ConvertPolygonMeshToTriangleMesh(None, output_vtk, dtype)
    return output_mesh
#%%
def FindDijkstraGraphGeodesicPath(mesh, start_node_idx, end_node_idx, mesh_vtk=None, dtype=None):
    if mesh.is_tri() == False:
        raise ValueError('only support triangle mesh')
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    finder=vtk.vtkDijkstraGraphGeodesicPath()
    finder.SetInputData(mesh_vtk)
    finder.SetStartVertex(start_node_idx)
    finder.SetEndVertex(end_node_idx)
    finder.Update()
    path_vtk=finder.GetOutput()
    path=torch.zeros((path_vtk.GetNumberOfPoints(), 3), dtype=dtype)
    for n in range(path_vtk.GetNumberOfPoints()):
        p=path_vtk.GetPoint(n)
        path[n,0]=p[0]
        path[n,1]=p[1]
        path[n,2]=p[2]
    node_idx_list=FindNearestNode(mesh, path)    
    return node_idx_list
#%%
def Subdivision(mesh, n_subdivisions, method='linear', mesh_vtk=None, dtype=None):
    #method: 
    # linear-> vtkLinearSubdivisionFilter
    # loop -> vtkLoopSubdivisionFilter
    # butterfly -> vtkButterflySubdivisionFilter
    if mesh_vtk is None:
        mesh_vtk=mesh.convert_to_vtk()
    if dtype is None:
        if mesh is not None:
            dtype=mesh.node.dtype
        else:
            raise ValueError('dtype is unknown')
    
    trifilter=vtk.vtkTriangleFilter()
    trifilter.SetInputData(mesh_vtk)
    trifilter.Update()
    mesh_vtk=trifilter.GetOutput()
    
    if method == 'linear':
        filter=vtk.vtkLinearSubdivisionFilter()
    elif method == 'loop':
        filter=vtk.vtkLoopSubdivisionFilter()
    elif method == 'butterfly':
        filter=vtk.vtkButterflySubdivisionFilter()
    else:
        raise ValueError
    filter.SetNumberOfSubdivisions(n_subdivisions)
    filter.SetInputData(mesh_vtk)
    filter.update()
    output_mesh=PolygonMesh()
    output_mesh.read_mesh_vtk(filter.GetOutput(), dtype)
    return output_mesh
