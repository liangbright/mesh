# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 19:52:22 2023

@author: liang
"""
import torch

def save_polyline_to_vtk(node, line, filename):
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