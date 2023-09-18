# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 22:35:58 2022

@author: liang
"""
from PolygonMesh import PolygonMesh
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
    mesh.build_node_adj_table()
    node_adj_table=mesh.node_adj_table
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
if __name__ == "__main__":
    filename="D:/MLFEA/TAA/data/343c1.5_fast/bav17_AortaModel_P0_best.pt"
    aorta=PolygonMesh()
    aorta.load_from_torch(filename)
    Curve=TracePolygonMeshBoundaryCurve(aorta, 0)