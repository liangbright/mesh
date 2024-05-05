# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:52:22 2023

@author: liang
"""
import numpy as np
import torch

def save_curve_as_vtk(curve_list, filename):
    if not isinstance(curve_list, list):
        raise ValueError('input curve_list must be a list')
    node, line=convert_curve_to_polyline(curve_list)
    save_polyline_as_vtk(node, line, filename)

def convert_curve_to_polyline(curve_list):
    #curve_list is a list of curves, each cure is a numpy array
    node=[]
    line=[]
    for k in range(0, len(curve_list)):
        curve=curve_list[k]        
        if isinstance(curve, torch.Tensor) or isinstance(curve, np.ndarray):
            curve=curve.tolist()
        if len(curve) > 0:
            idx_start=len(node)
            node.extend(curve)
            line.append(np.arange(idx_start, idx_start+len(curve)).tolist())
    node=torch.tensor(node, dtype=torch.float64)
    return node, line
    
def save_polyline_as_vtk(node, line, filename):
    #node: (N, 3), 3D positions
    #line[k] is the k-th polyline, a list of node indexes
    try:
        temp=line[0][0]
    except:
        raise ValueError("line needs to be a nested list")

    out=[]
    out.append('# vtk DataFile Version 4.2'+'\n')
    out.append('vtk output'+'\n')
    out.append('ASCII'+'\n')
    out.append('DATASET POLYDATA'+'\n')
    #------------------------------------------------------------------
    out.append('POINTS '+str(node.shape[0])+' double'+'\n')
    if isinstance(node, torch.Tensor):
        node=node.detach().to('cpu')
    for n in range(0, node.shape[0]):
        x=float(node[n,0])
        y=float(node[n,1])
        z=float(node[n,2])
        out.append(str(x)+' '+str(y)+' '+str(z)+'\n')
    #------------------------------------------------------------------
    if isinstance(line, torch.Tensor):
        line=line.detach().to('cpu')
    offset_count=0
    for m in range(0, len(line)):
        offset_count+=1+len(line[m])
    out.append('LINES '+str(len(line))+' '+str(offset_count)+'\n')
    for m in range(0, len(line)):
        text=str(len(line[m]))
        for k in range(0, len(line[m])):
            id=int(line[m][k])
            text=text+' '+str(id)
        out.append(text+'\n')
    #------------------------------------------------------------------
    with open(filename, 'w', encoding = 'utf-8') as file:
        file.writelines(out)