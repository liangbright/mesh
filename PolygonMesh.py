# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import torch
import numpy as np
_Flag_VTK_IMPORT_=False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class PolygonMesh:

    def __init__(self):
        self.node=[]#shape: Nx3
        self.element=[]# list or tensor, len()=M
        self.node_name_to_index={}
        self.element_name_to_index={}
        self.node_set={}
        self.element_set={}
        self.adj_node_link=[]# N x 2
        self.adj_element_link=[]# M x 2, share an edge
        self.node_to_element_link=[]# (~N+M)x2
        self.node_data={}
        self.element_data={}
        self.mesh_data={}

    def build_adj_node_link(self):
        adj_node_link=[]
        for n in range(0, len(self.element)):
            e=self.element[n]
            for m in range(0, len(e)):
                if m < len(e)-1:
                    adj_node_link.append([e[m], e[m+1]])
                    adj_node_link.append([e[m+1], e[m]])
                else:
                    adj_node_link.append([e[m], e[0]])
                    adj_node_link.append([e[0], e[m]])
        adj_node_link=torch.tensor(adj_node_link, dtype=torch.int64)
        adj_node_link=torch.unique(adj_node_link, dim=0, sorted=True)
        self.adj_node_link=adj_node_link

    def build_adj_element_link(self):
        adj_element_link=[]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for n in range(0, len(element)):
            e_n=element[n]
            for m in range(n+1, len(element)):
                e_m=element[m]
                temp=np.isin(e_n, e_m, assume_unique=True)
                temp=np.sum(temp)
                if temp == 2:
                    adj_element_link.append([n, m])
                    adj_element_link.append([m, n])
        adj_element_link=torch.tensor(adj_element_link, dtype=torch.int64)
        adj_element_link=torch.unique(adj_element_link, dim=0, sorted=True)
        self.adj_element_link=adj_element_link

    def load_from_vtk(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot import vtk")
            return
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        node=np.zeros((mesh_vtk.GetNumberOfPoints(), 3))
        for n in range(mesh_vtk.GetNumberOfPoints()):
            node[n]=mesh_vtk.GetPoint(n)
        element=[]
        m_list=[]
        for n in range(mesh_vtk.GetNumberOfCells()):
            cell_n=mesh_vtk.GetCell(n)
            m_list.append(cell_n.GetNumberOfPoints())
            temp=[]
            for k in range(cell_n.GetNumberOfPoints()):
                temp.append(cell_n.GetPointId(k))
            element.append(temp)
        self.node=torch.tensor(node, dtype=dtype)
        self.element=element
        if min(m_list) == max(m_list):
            self.element=torch.tensor(element, dtype=torch.int64)
        #---------- load PointData -----------#
        PointDataSetCount = mesh_vtk.GetPointData().GetNumberOfArrays()
        for n in range(0, PointDataSetCount):
            vtk_array=mesh_vtk.GetPointData().GetArray(n)
            name=vtk_array.GetName()
            data=np.zeros((vtk_array.GetNumberOfTuples(), vtk_array.GetNumberOfComponents()))
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    data[i,j]=vtk_array.GetComponent(i,j)
            self.node_data[name]=torch.tensor(data, dtype=dtype)
        #---------- load CellData -----------#
        CellDataSetCount = mesh_vtk.GetCellData().GetNumberOfArrays()
        for n in range(0, CellDataSetCount):
            vtk_array=mesh_vtk.GetCellData().GetArray(n)
            name=vtk_array.GetName()
            data=np.zeros((vtk_array.GetNumberOfTuples(), vtk_array.GetNumberOfComponents()))
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    data[i,j]=vtk_array.GetComponent(i,j)
            self.element_data[name]=torch.tensor(data, dtype=dtype)

    def convert_to_vtk(self):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot import vtk")
            return
        Points_vtk = vtk.vtkPoints()
        Points_vtk.SetDataTypeToDouble()
        Points_vtk.SetNumberOfPoints(len(self.node))
        for n in range(0, len(self.node)):
            Points_vtk.SetPoint(n, float(self.node[n,0]), float(self.node[n,1]), float(self.node[n,2]))
        mesh_vtk = vtk.vtkPolyData()
        mesh_vtk.SetPoints(Points_vtk)
        mesh_vtk.Allocate(len(self.element))
        for n in range(0, len(self.element)):
            e=list(self.element[n])
            if len(e) <= 2:
                print('len(e)=', len(e), ': ignored')
                continue
            elif len(e) == 3:
                cell_type=vtk.VTK_TRIANGLE
            elif len(e) == 4:
                cell_type=vtk.VTK_QUAD
            else:
                cell_type=vtk.VTK_POLYGON
            mesh_vtk.InsertNextCell(cell_type, len(e), e)
        #--------- convert node_data to PointData --------#
        for name, data in self.node_data.items():
            #data should be a 2D array (self.node.shape[0], ?)
            if self.node.shape[0] != data.shape[0]:
                raise ValueError("self.node.shape[0] != data.shape[0]:"+name)
            vtk_array=vtk.vtkDoubleArray()
            #run SetNumberOfComponents before SetNumberOfTuples
            vtk_array.SetNumberOfComponents(data.shape[1])
            vtk_array.SetNumberOfTuples(data.shape[0])
            vtk_array.SetName(name)
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    vtk_array.SetComponent(i,j,float(data[i,j]))
            mesh_vtk.GetPointData().AddArray(vtk_array)
        #--------- convert element_data to CellData --------#
        for name, data in self.element_data.items():
            #data should be a 2D array (len(self.element), ?)
            if len(self.element) != data.shape[0]:
                raise ValueError("len(self.element) != data.shape[0]:"+name)
            vtk_array=vtk.vtkDoubleArray()
            vtk_array.SetNumberOfComponents(data.shape[1])
            vtk_array.SetNumberOfTuples(data.shape[0])
            vtk_array.SetName(name)
            for i in range(0, data.shape[0]):
                for j in range(0, data.shape[1]):
                    vtk_array.SetComponent(i,j,float(data[i,j]))
            mesh_vtk.GetCellData().AddArray(vtk_array)
        return mesh_vtk

    def save_by_vtk(self, filename):
        mesh_vtk=self.convert_to_vtk()
        if mesh_vtk is None:
            return
        writer=vtk.vtkPolyDataWriter()
        writer.SetFileTypeToASCII()
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        writer.Write()

    def save_by_torch(self, filename):
        torch.save({"node":self.node,
                    "element":self.element,
                    "node_data":self.node_data,
                    "element_data":self.element_data,
                    "mesh_data":self.mesh_data},
                   filename)

    def load_from_torch(self, filename):
        data=torch.load(filename, map_location="cpu")
        if "node" in data.keys():
            self.node=data["node"]
        else:
            raise ValueError("node is not in data.keys()")
        if "element" in data.keys():
            self.element=data["element"]
        else:
            raise ValueError("element is not in data.keys()")
        if "node_data" in data.keys():
            self.node_data=data["node_data"]
        if "element_data" in data.keys():
            self.element_data=data["element_data"]
        if "mesh_data" in data.keys():
            self.mesh_data=data["mesh_data"]
#%%
if __name__ == "__main__":
    filename="C:/Research/AorticValve/TAVR/1908788_0_im_5_phase1/OutputWall_2017_5_17_10_28.json.vtk"
    root=PolygonMesh()
    root.load_from_vtk(filename, torch.float32)
    root.save_by_vtk("test_poly.vtk")
