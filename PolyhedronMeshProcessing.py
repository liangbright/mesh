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
def ExtractSurface():
    pass