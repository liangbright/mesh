    # -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 22:24:13 2021

@author: liang
"""
import numpy as np
import torch
_Flag_VTK_IMPORT_ = False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class PolyhedronMesh:

    def __init__(self):
        self.node=[]
        self.element=[]
        self.node_name_to_index={}
        self.element_set={}
        self.node_data={}
        self.element_data={}
        self.mesh_data={}

    def load_from_vtk(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot import vtk")
            return
        reader = vtk.vtkUnstructuredGridReader()
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
            print("cannot save vtk")
            return
        Points_vtk = vtk.vtkPoints()
        Points_vtk.SetDataTypeToDouble()
        Points_vtk.SetNumberOfPoints(len(self.node))
        for n in range(0, len(self.node)):
            Points_vtk.SetPoint(n, float(self.node[n,0]), float(self.node[n,1]), float(self.node[n,2]))
        mesh_vtk = vtk.vtkUnstructuredGrid()
        mesh_vtk.SetPoints(Points_vtk)
        mesh_vtk.Allocate(len(self.element))
        for n in range(0, len(self.element)):
            e=list(self.element[n])
            if len(e) == 8:
                cell_type=vtk.VTK_HEXAHEDRON
            elif len(e) == 6:
                cell_type=vtk.VTK_WEDGE
            elif len(e) == 4:
                cell_type=vtk.VTK_TETRA
            else:
                cell_type=vtk.VTK_POLYHEDRON
            mesh_vtk.InsertNextCell(cell_type, len(e), e)
        #--------- convert node_data to PointData --------#
        for name, data in self.node_data.items():
            #data should be a 2D array (self.node.shape[0], ?)
            if self.node.shape[0] != data.shape[0]:
                raise ValueError("self.node.shape[0] != data.shape[0], name:"+name)
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
                raise ValueError("len(self.element) != data.shape[0], name:"+name)
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
        writer=vtk.vtkUnstructuredGridWriter()
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
def get_sub_mesh_hex8(node, element, element_sub):
    #element.shape (M,8)
    node_idlist, element_out=torch.unique(element_sub.reshape(-1), return_inverse=True)
    node_out=node[node_idlist]
    element_out=element_out.view(-1,8)
    return node_out, element_out
#%%
if __name__ == "__main__":
    #%%
    filename="C:/Research/MLFEA/TAVR/FE/1908788_0_im_5_phase1_Root_solid_three_layers_aligned.vtk"
    root=PolyhedronMesh()
    root.load_from_vtk(filename)
    #%%
    root.node_name_to_index["C01"]=131
    root.node_name_to_index["C12"]=2010
    root.node_name_to_index["C20"]=1
    root.node_name_to_index["H0"]=2
    root.node_name_to_index["H1"]=1883
    root.node_name_to_index["H2"]=3759
    root.element_set['Leaflet1']=np.arange(0, 888)
    root.element_set['Leaflet2']=np.arange(888, 2*888)
    root.element_set['Leaflet3']=np.arange(2*888, 3*888)
    #%%
    node_stress=np.random.rand(root.node.shape[0], 9)
    root.node_data["stress"]=node_stress
    element_stress=np.random.rand(len(root.element), 9)
    root.element_data["stress"]=element_stress
    #%%
    root.update_element_normal()
    #%%
    root.save_by_vtk("test.vtk")
    #%%
    root=PolyhedronMesh()
    root.load_from_vtk("test.vtk")
