# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:07:37 2022

@author: liang
"""
import torch
import numpy as np
from SaveMeshAsVTKFile import save_polygon_mesh_to_vtk, save_polyhedron_mesh_to_vtk
_Flag_VTK_IMPORT_=False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class Mesh:
    def __init__(self, mesh_type):
        if ('polyhedron' in mesh_type) or ('polygon' in mesh_type):
            pass
        else:
            raise ValueError('unknown mesh_type:'+mesh_type)
        self.mesh_type=mesh_type
        self.node=[] #Nx2 or Nx3
        self.element=[] #[e_1,...e_M], e1 is a list of node indexes in e1
        self.element_type=None#None means all elements have the same type
        self.node_name_to_index={} #e.g., {'landmark1':10}
        self.node_set={} #e.g., {'set1':[0,1,2]}
        self.element_set={} #e.g., {'set1':[1,3,5]}
        self.node_data={} #e.g., {'stress':stress}, stress is Nx9 2D array
        self.element_data={} #e.g., {'stress':stress}, stress is Mx9 2D array
        self.mesh_data={} # it is only saved by torch
        self.node_to_element_link=None
        self.adj_node_link={'directed':None, 'undirected':None}
        self.adj_element_link=None

    def load_from_vtk(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot import vtk")
            return
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                ValueError('unknown dtype:'+str(dtype))
        if 'polyhedron' in self.mesh_type:
            reader = vtk.vtkUnstructuredGridReader()
        elif 'polygon' in self.mesh_type:
            reader = vtk.vtkPolyDataReader()
        else:
            raise ValueError('unknown mesh_type:'+self.mesh_type)
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

    @staticmethod
    def get_vtk_cell_type(element_type, n_nodes):
        if 'polyhedron' in element_type:
            if n_nodes == 4:
                cell_type=vtk.VTK_TETRA
            elif n_nodes == 6:
                cell_type=vtk.VTK_WEDGE
            elif n_nodes == 8:
                cell_type=vtk.VTK_HEXAHEDRON
            else:
                cell_type=vtk.VTK_POLYHEDRON
        elif 'polygon' in element_type:
            if n_nodes == 3:
                cell_type=vtk.VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type=vtk.VTK_QUAD
            else:
                cell_type=vtk.VTK_POLYGON
        else:
            raise ValueError('unknown element_type:'+element_type)
        return cell_type

    def convert_to_vtk(self):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot save vtk")
            return
        Points_vtk = vtk.vtkPoints()
        Points_vtk.SetDataTypeToDouble()
        Points_vtk.SetNumberOfPoints(len(self.node))
        for n in range(0, len(self.node)):
            Points_vtk.SetPoint(n, float(self.node[n,0]), float(self.node[n,1]), float(self.node[n,2]))
        if 'polyhedron' in self.mesh_type:
            mesh_vtk = vtk.vtkUnstructuredGrid()
        elif 'polygon' in self.mesh_type:
            mesh_vtk = vtk.vtkPolyData()
        else:
            raise ValueError('unknown mesh_type:'+self.mesh_type)
        mesh_vtk.SetPoints(Points_vtk)
        mesh_vtk.Allocate(len(self.element))
        for n in range(0, len(self.element)):
            e=[int(id) for id in self.element[n]]
            if self.element_type is None:
                cell_type=Mesh.get_vtk_cell_type(self.mesh_type, len(e))
            else:
                cell_type=Mesh.get_vtk_cell_type(self.element_type[n], len(e))
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

    def save_by_vtk(self, filename, ascii=True, vtk42=True):
        if _Flag_VTK_IMPORT_ == False:
            if 'polyhedron' in self.mesh_type:
                save_polyhedron_mesh_to_vtk(self, filename)
            elif 'polygon' in self.mesh_type:
                save_polygon_mesh_to_vtk(self, filename)
            else:
                raise ValueError('unknown mesh_type:'+self.mesh_type)
            return
        #-----------------------------
        mesh_vtk=self.convert_to_vtk()
        if mesh_vtk is None:
            return
        if 'polyhedron' in self.mesh_type:
            writer=vtk.vtkUnstructuredGridWriter()
        elif 'polygon' in self.mesh_type:
            writer=vtk.vtkPolyDataWriter()
        else:
            raise ValueError('unknown mesh_type:'+self.mesh_type)
        if vtk42 == True:
            try:
                version=[int(x) for x in vtk.__version__ .split('.')]
                if version[0]>=9 and version[1]>=1:
                    writer.SetFileVersion(42)
                else:
                    print('cannot save to 4.2 vtk version')
            except:
                print('cannot save to 4.2 vtk version')
        if ascii == True:
            writer.SetFileTypeToASCII()
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        writer.Write()

    def save_by_torch(self, filename):
        torch.save({"mesh_type":self.mesh_type,
                    "node":self.node,
                    "element":self.element,
                    "node_set":self.node_set,
                    "element_set":self.element_set,
                    "node_data":self.node_data,
                    "element_data":self.element_data,
                    "mesh_data":self.mesh_data,
                    "node_name_to_index":self.node_name_to_index,
                    "node_to_element_link":self.node_to_element_link,
                    "adj_node_link":self.adj_node_link,
                    "adj_element_link":self.adj_element_link},
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
        if "node_set" in data.keys():
            self.node_set=data["node_set"]
        if "element_set" in data.keys():
            self.element_set=data["element_set"]
        if "node_data" in data.keys():
            self.node_data=data["node_data"]
        if "element_data" in data.keys():
            self.element_data=data["element_data"]
        if "mesh_data" in data.keys():
            self.mesh_data=data["mesh_data"]
        if "node_name_to_index" in data.keys():
            self.node_name_to_index=data["node_name_to_index"]
        if "node_to_element_link" in data.keys():
            self.node_to_element_link=data["node_to_element_link"]
        if "adj_node_link" in data.keys():
            self.adj_node_link=data["adj_node_link"]
        if "adj_element_link" in data.keys():
            self.adj_element_link=data["adj_element_link"]

    def build_adj_node_link(self):
        #assume: nodes in an element are adjacent to each other
        adj_node_link=[]
        for m in range(0, len(self.element)):
            e=self.element[m]
            for j in range(0, len(e)):
                for i in range(0, len(e)):
                    if i != j:
                        adj_node_link.append([e[j], e[i]])
                        adj_node_link.append([e[i], e[j]])
        adj_node_link=torch.tensor(adj_node_link, dtype=torch.int64)
        adj_node_link=torch.unique(adj_node_link, dim=0, sorted=True)
        self.adj_node_link['undirected']=adj_node_link

    def build_node_to_element_link(self):
        #do not do this: node_to_element_link=[[]]*self.node.shape[0]
        # a=[[]]*2=[[],[]], and a[0] and a[1] are the same object
        # a=[[],[]], and a[0] and a[1] are two different objects
        node_to_element_link=[[] for _ in range(self.node.shape[0])]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(element)):
            e=element[m]
            for k in range(0, len(e)):
                node_to_element_link[e[k]].append(m)
        self.node_to_element_link=node_to_element_link

    def build_adj_element_link_n1(self):
        self.build_node_to_element_link()
        adj_element_link=[]
        for n in range(0, len(self.node_to_element_link)):
            e_set=self.node_to_element_link[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    adj_element_link.append([eid1, eid2])
                    adj_element_link.append([eid2, eid1])
        adj_element_link=torch.tensor(adj_element_link, dtype=torch.int64)
        adj_element_link=torch.unique(adj_element_link, dim=0, sorted=True)
        self.adj_element_link=adj_element_link

    def build_adj_element_link(self, min_shared_nodes=1):
        if min_shared_nodes == 1:
            return self.build_adj_element_link_n1()
        #---------the code is very slow -------------------
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
                if temp >= min_shared_nodes:
                    adj_element_link.append([n, m])
                    adj_element_link.append([m, n])
        adj_element_link=torch.tensor(adj_element_link, dtype=torch.int64)
        adj_element_link=torch.unique(adj_element_link, dim=0, sorted=True)
        self.adj_element_link=adj_element_link
#%%
if __name__ == "__main__":
    #%%
    filename="D:/MLFEA/TAVR/FE/1908788_0_im_5_phase1_Root_solid_three_layers_aligned.vtk"
    root1=Mesh('polyhedron')
    root1.load_from_vtk(filename, dtype=torch.float32)
    root1.build_node_to_element_link()
    #%%
    root1.save_by_vtk("test.vtk")
    #%%
    root2=Mesh('polyhedron')
    root2.load_from_vtk("test.vtk", dtype=torch.float32)
    #%%
    import time
    t1=time.time()
    root2.build_node_to_element_link()
    t2=time.time()
    print('t2-t1', t2-t1)
    #%%
    import time
    t1=time.time()
    root2.build_adj_element_link()
    t2=time.time()
    print('t2-t1', t2-t1)
