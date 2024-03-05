# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 00:11:56 2023

@author: liang
"""
import torch
import torch_scatter
import numpy as np
from PolyhedronMesh import PolyhedronMesh
from TetrahedronMesh import TetrahedronMesh
from HexahedronMesh import HexahedronMesh
from Tet10Mesh import Tet10Mesh
from copy import deepcopy
from MeshProcessing import SimpleSmoother, SimpleSmootherForMesh, ComputeAngleBetweenTwoVectorIn3D, TracePolyline, \
                            IsCurveClosed, MergeMesh, FindConnectedRegion, FindNearestNode
try:
    import vtk
except:
    print("cannot import vtk")
#%%
def ExtractSurface(mesh):
    #extract surface and make sure surface normal is from inside to outside
    if ((not isinstance(mesh, TetrahedronMesh)) 
        and (not isinstance(mesh, Tet10Mesh))
        and (not isinstance(mesh, HexahedronMesh))):
        print('mesh is ', type(mesh))
        raise NotImplementedError
    face_idx_list=mesh.find_boundary_face()
    adj_table=mesh.face_to_element_adj_table    
    Surface=[]
    for n in range(0, len(face_idx_list)):
        face_idx=face_idx_list[n]
        fa=mesh.face[face_idx]
        if not isinstance(fa, list):
            fa=fa.tolist()
        fa=np.array([fa], dtype=np.int64)
        fa=np.sort(fa, axis=1)
        elm_idx=adj_table[face_idx][0]
        elm=mesh.element[elm_idx]
        if not isinstance(elm, list):
            elm=elm.tolist()
        face=[]
        if isinstance(mesh, TetrahedronMesh):
            id0, id1, id2, id3=elm
            face.append([id0, id2, id1])
            face.append([id0, id1, id3])
            face.append([id0, id3, id2])
            face.append([id1, id2, id3])
        elif isinstance(mesh, Tet10Mesh):
            id0, id1, id2, id3, id4, id5, id6, id7, id8, id9=elm
            face.append([id0, id6, id2, id5, id1, id4])
            face.append([id0, id4, id1, id8, id3, id7])
            face.append([id0, id7, id3, id9, id2, id6])
        elif isinstance(mesh, HexahedronMesh):
            id0, id1, id2, id3, id4, id5, id6, id7=elm
            face.append([id0, id3, id2, id1])
            face.append([id4, id5, id6, id7])
            face.append([id0, id1, id5, id4])
            face.append([id1, id2, id6, id5])
            face.append([id2, id3, id7, id6])
            face.append([id3, id0, id4, id7])
        else:
            raise ValueError
        face=np.array(face, dtype=np.int64)
        face_sorted=np.sort(face, axis=1)
        best_idx=np.abs((face_sorted-fa)).sum(axis=1).argmin().item()
        Surface.append(face[best_idx].tolist())
    return Surface
            