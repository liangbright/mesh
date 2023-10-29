# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 22:29:00 2023

@author: liang
"""
import numpy as np

def read_abaqus_inp(filename, remove_unused_node):
    with open(filename, 'r') as file:
        inp=file.readlines()
        print('total number of lines in inp:', len(inp))
    #-----------
    node, node_id=read_node(inp)
    print('total number of nodes:', len(node))
    element, element_id, element_type, element_set=read_element(inp)
    print('total number of elements:', len(element))
    #-----------
    node=np.array(node)
    node_id=np.array(node_id)
    element=np.array(element, dtype='object')
    element_id=np.array(element_id)
    element_type=np.array(element_type, dtype='object')
    #-----------
    node, node_id, element=clean_data_re_label_node(node, node_id, element)
    if remove_unused_node == True:
        node, node_id, element=clean_data_remove_unused_node(node, node_id, element)
    #-----------
    element=element.tolist()
    return node, node_id, element, element_id, element_type, element_set
#%%
def read_node(inp):
    node=[]
    node_id=[]
    lineindex=-1
    for k in range(0, len(inp)):
        if "*node" in inp[k].lower():
            lineindex=k
            break
    if lineindex < 0:
        print('node keyword is not found')
        return node, node_id
    k=lineindex
    while True:
        k=k+1
        if k >= len(inp):
            break
        if "*" in inp[k]:
            break
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        node_id.append(int(temp[0]))
        node.append([float(temp[1]), float(temp[2]), float(temp[3])])
    return node, node_id
#%%
def read_element(inp):
    element=[]
    element_id=[]
    element_type=[]
    element_set={}
    lineindexlist=[]
    for k in range(0, len(inp)):
        if "*element" in inp[k].lower():
            lineindexlist.append(k)
    if len(lineindexlist) == 0:
        print('element keyword is not found')
        return element, element_id, element_set
    for lineindex in lineindexlist:
        k=lineindex
        eltype=None
        temp=inp[k].replace(" ", "")
        temp=temp.split(",")
        for v in temp:
            if "type" in v.lower():
                eltype=v.split("=")[-1]
                eltype=eltype.replace("\n", "")
                eltype=eltype.replace(" ", "")
                break
        elset=None
        for v in temp:
            if "elset" in v.lower():
                elset=v.split('=')[-1]
                elset=elset.replace("\n", "")
                break
        if elset is not None:
            if elset not in element_set.keys():
                element_set[elset]=[]

        while True:
            k=k+1
            if k >= len(inp):
                break
            if "*" in inp[k]:
                break
            temp=inp[k].replace(" ", "")
            temp=temp.split(",")
            temp=[int(a) for a in temp]
            element_id.append(int(temp[0]))
            element.append(temp[1:])
            element_type.append(eltype)
            if elset is not None:
                element_set[elset].append(len(element)-1)#len(element)-1 is the index of the current element
    return element, element_id, element_type, element_set
#%%
def clean_data_re_label_node(node, node_id, element):
    #node_id could start from an arbitrary number, e.g, 1000
    #sort node by id
    idx_sorted=np.argsort(node_id)
    node=node[idx_sorted]
    node_id=node_id[idx_sorted]
    #map node id to index in node
    map={}
    for n in range(0, len(node)):
        map[node_id[n]]=n
    for m in range(0, len(element)):
        for n in range(0, len(element[m])):
            element[m][n]=map[element[m][n]]
    return node, node_id, element
#%%
def clean_data_remove_unused_node(node, node_id, element):
    used_old_idx_list=[]
    for m in range(0, len(element)):
        used_old_idx_list.extend(element[m])
    used_old_idx_list=np.unique(used_old_idx_list)
    num_unused_nodes=len(node)-len(used_old_idx_list)
    print('total number of unused nodes:', num_unused_nodes)
    if num_unused_nodes == 0:
        return node, node_id, element
    #map old node id to new node id
    map={}
    for n in range(0, len(used_old_idx_list)):
        old_idx=used_old_idx_list[n]
        map[old_idx]=n # n is new_idx
    for m in range(0, len(element)):
        for n in range(0, len(element[m])):
            element[m][n]=map[element[m][n]]
    node=node[used_old_idx_list]
    node_id=node_id[used_old_idx_list]
    return node, node_id, element