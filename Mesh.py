# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:07:37 2022

@author: liang
"""
import torch
#from torch_sparse import SparseTensor
import numpy as np
from copy import deepcopy
from SaveMeshAsVTKFile import (save_polyline_as_vtk, 
                               save_polygon_mesh_to_vtk, 
                               save_polyhedron_mesh_to_vtk)
import os
_Flag_VTK_IMPORT_=False
try:
    import vtk
    _Flag_VTK_IMPORT_=True
except:
    print("cannot import vtk")
#%%
class Mesh:
    def __init__(self, node, element, element_type, mesh_type):
        if (('polyhedron' not in mesh_type)
            and ('polygon' not in mesh_type)
            and ('polyline' not in mesh_type)):
            raise ValueError('unknown mesh_type: '+mesh_type)
        #--------------------------------------------------------------------
        self.name="" # name of the mesh
        self.mesh_type=mesh_type
        self.node=[] #Nx3
        self.element=[] #[e_1,...e_M], e1 is a list of node indexes in e1
        self.element_type=None #None then element_type will be infered by using mesh_type and node count per element
        self.mapping_from_node_name_to_index={} #e.g., {'landmark1':10}
        self.node_set={} #e.g., {'set1':[0,1,2]}
        self.element_set={} #e.g., {'set1':[1,3,5]}
        self.node_data={} #e.g., {'stress':stress}, stress is Nx9 2D array
        self.element_data={} #e.g., {'stress':stress}, stress is Mx9 2D array
        self.mesh_data={} # it is only saved by torch
        #--------------------------------------------------------------------
        self.clear_adj_info() #initialize adj info to None
        #--------------------------------------------------------------------
        if node is not None:
            if isinstance(node, list):
                node=torch.tensor(node, dtype=torch.float32)
            elif isinstance(node, np.ndarray):
                if node.dtype == np.float64:
                    node=torch.tensor(node, dtype=torch.float64)
                else:
                    node=torch.tensor(node, dtype=torch.float32)            
            self.node=node
        #--------------------------------------------------------------------
        if element is not None:
            if isinstance(element, list) or isinstance(node, np.ndarray):
                try:
                    element=torch.tensor(element, dtype=torch.int64)
                except:
                    pass
            elif isinstance(element, torch.Tensor):
                  pass
            else:
                raise ValueError("unsupported python-object type of element")
            self.element=element
        #--------------------------------------------------------------------
        if element_type is not None:
            if isinstance(element_type, np.ndarray):
                pass
            elif isinstance(element_type, list):
                element_type=np.ndarray(element_type)
            elif isinstance(element, torch.Tensor):
                  element_type=element_type.cpu().numpy()
            else:
                raise ValueError("unsupported python-object type of element_type")
            self.element_type=element_type
        #--------------------------------------------------------------------

    def clear_adj_info(self):
        #if a new node or a new element is added/deleted after a mesh is created,
        #then the adj info will become invalid and therefore need to be cleared
        self.edge=None #only one undirected edge between two adj nodes
        self.map_node_pair_to_edge=None#sparse matrix: self.map_node_pair_to_edge[i,j] - 1 is index (>=0) in self.edge
        self.node_to_node_adj_link=None#similar to edge_index in pytorch geometric
        self.node_to_node_adj_table=None
        self.node_to_edge_adj_table=None
        self.node_to_element_adj_table=None
        self.edge_to_edge_adj_table=None
        self.edge_to_element_adj_table=None# an edge of an element
        self.element_to_edge_adj_table=None# an edge of an element
        if 'polygon' in self.mesh_type:
            self.element_to_element_adj_link={"node":None, "edge":None}
            self.element_to_element_adj_table={"node":None, "edge":None}
        elif 'polyhedron' in self.mesh_type:
            self.face=None
            self.face_to_element_adj_table=None
            self.element_to_face_adj_table=None
            self.element_to_element_adj_link={"node":None, "edge":None, "face":None}
            self.element_to_element_adj_table={"node":None, "edge":None, "face":None}

    def load_from_stl(self, filename, dtype):
        if not os.path.isfile(filename):
            raise ValueError("not exist: "+filename)
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError("vtk is not imported")
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        if 'polygon' in self.mesh_type:
            reader = vtk.vtkSTLReader()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        self.read_mesh_vtk(mesh_vtk, dtype)
    
    def load_from_vtp(self, filename, dtype):
        if not os.path.isfile(filename):
            raise ValueError("not exist: "+filename)
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError("vtk is not imported")
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        if 'polygon' in self.mesh_type:
            reader = vtk.vtkXMLPolyDataReader()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        try:
            self.read_mesh_vtk(mesh_vtk, dtype)
        except:
            raise ValueError('cannot convert vtk data to mesh')

    def load_from_vtk(self, filename, dtype):
        if not os.path.isfile(filename):
            raise ValueError("not exist: "+filename)
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError("vtk is not imported")
        if isinstance(dtype, str):
            if dtype == 'float32':
                dtype=torch.float32
            elif dtype == 'float64':
                dtype=torch.float64
            else:
                raise ValueError('unsupported dtype:'+str(dtype))
        if 'polyhedron' in self.mesh_type:
            reader = vtk.vtkUnstructuredGridReader()
        elif ('polygon' in self.mesh_type) or ('polyline' in self.mesh_type):
            reader = vtk.vtkPolyDataReader()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        reader.SetFileName(filename)
        reader.Update()
        mesh_vtk = reader.GetOutput()
        try:
            self.read_mesh_vtk(mesh_vtk, dtype)
        except:
            raise ValueError('cannot convert vtk data to mesh')

    def read_mesh_vtk(self, mesh_vtk, dtype):
        node=np.zeros((mesh_vtk.GetNumberOfPoints(), 3))
        for n in range(mesh_vtk.GetNumberOfPoints()):
            node[n]=mesh_vtk.GetPoint(n)
        if len(node) == 0:
            raise ValueError('cannot load node')
        element=[]
        m_list=[]
        for n in range(mesh_vtk.GetNumberOfCells()):
            cell_n=mesh_vtk.GetCell(n)
            m_list.append(cell_n.GetNumberOfPoints())
            temp=[]
            for k in range(cell_n.GetNumberOfPoints()):
                temp.append(cell_n.GetPointId(k))
            element.append(temp)
        if len(element) == 0:
            raise ValueError('cannot load element')
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
    def get_vtk_cell_type(mesh_type, n_nodes):
        if 'polyhedron' in mesh_type:
            if n_nodes == 4:
                cell_type=vtk.VTK_TETRA
            elif n_nodes == 6:
                cell_type=vtk.VTK_WEDGE
            elif n_nodes == 8:
                cell_type=vtk.VTK_HEXAHEDRON
            elif n_nodes == 10:
                cell_type=vtk.VTK_QUADRATIC_TETRA
            else:
                cell_type=vtk.VTK_POLYHEDRON
        elif 'polygon' in mesh_type:
            if n_nodes == 3:
                cell_type=vtk.VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type=vtk.VTK_QUAD
            elif n_nodes == 6:
                cell_type=vtk.VTK_QUADRATIC_TRIANGLE
            else:
                cell_type=vtk.VTK_POLYGON
        elif 'polyline' in mesh_type:
            cell_type=vtk.VTK_POLY_LINE
        else:
            raise ValueError('unsupported mesh_type:'+mesh_type)
        return cell_type

    @staticmethod
    def get_vtk_cell_type_from_element_type(element_type):
        #element_type: tri3, quad4, hex8, etc
        raise NotImplementedError

    def convert_to_vtk(self):
        if _Flag_VTK_IMPORT_ == False:
            raise ValueError("vtk is not imported")
        Points_vtk = vtk.vtkPoints()
        Points_vtk.SetDataTypeToDouble()
        Points_vtk.SetNumberOfPoints(len(self.node))
        for n in range(0, len(self.node)):
            Points_vtk.SetPoint(n, float(self.node[n,0]), float(self.node[n,1]), float(self.node[n,2]))
        if 'polyhedron' in self.mesh_type:
            mesh_vtk = vtk.vtkUnstructuredGrid()
        elif ('polygon' in self.mesh_type) or 'polyline' in self.mesh_type:
            mesh_vtk = vtk.vtkPolyData()
        else:
            raise ValueError('unsupported mesh_type:'+self.mesh_type)
        mesh_vtk.SetPoints(Points_vtk)
        mesh_vtk.Allocate(len(self.element))
        for n in range(0, len(self.element)):
            e=[int(id) for id in self.element[n]]
            if self.element_type is None:
                cell_type=Mesh.get_vtk_cell_type(self.mesh_type, len(e))
            else:
                cell_type=Mesh.translate_element_type_to_vtk_cell_type(self.element_type[n])
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

    def save_as_vtk(self, filename, vtk42=True, use_vtk=False):
        if _Flag_VTK_IMPORT_ == False or use_vtk == False:
            if 'polyhedron' in self.mesh_type:
                save_polyhedron_mesh_to_vtk(self, filename)
                if vtk42 == False:
                    print("Mesh save_as_vtk: can only save to 4.2 version vtk, although vtk42=False")
            elif 'polygon' in self.mesh_type:
                save_polygon_mesh_to_vtk(self, filename)
                if vtk42 == False:
                    print("Mesh save_as_vtk: can only save to 4.2 version vtk, although vtk42=False")
            elif 'polyline' in self.mesh_type:
                save_polyline_as_vtk(self, filename)
            else:
                raise ValueError('unsupported mesh_type: '+self.mesh_type)
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
            raise ValueError('unsupported mesh_type: '+self.mesh_type)
        if vtk42 == True:
            version=vtk.vtkVersion.GetVTKVersion()
            version=[int(a) for a in version.split('.')]
            if version[0]<=8:
                pass
            elif version[0]>=9 and version[1]>=1:
                writer.SetFileVersion(42)
            else:
                print('Mesh save_as_vtk: cannot save to 4.2 version vtk')
        writer.SetFileTypeToASCII()
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        writer.Write()
    
    def save_as_vtp(self, filename):
        if _Flag_VTK_IMPORT_ == False:            
            raise ValueError('vtk is not imported')
        #-----------------------------
        mesh_vtk=self.convert_to_vtk()
        if mesh_vtk is None:
            return
        if 'polygon' in self.mesh_type:
            writer=vtk.vtkXMLPolyDataWriter()
        else:
            raise ValueError('unsupported mesh_type: '+self.mesh_type)
        writer.SetInputData(mesh_vtk)
        writer.SetFileName(filename)
        writer.Write()        

    def save_as_torch(self, filename, save_adj_info=True):
        data={"mesh_type":self.mesh_type,
              "node":self.node,
              "element":self.element,
              "node_set":self.node_set,
              "element_set":self.element_set,
              "node_data":self.node_data,
              "element_data":self.element_data,
              "mesh_data":self.mesh_data,
              "mapping_from_node_name_to_index":self.mapping_from_node_name_to_index}
        if save_adj_info == True:
            data["edge"]=self.edge
            data["node_to_node_adj_link"]=self.node_to_node_adj_link
            data["node_to_node_adj_table"]=self.node_to_node_adj_table
            data["node_to_edge_adj_table"]=self.node_to_edge_adj_table
            data["node_to_element_adj_table"]=self.node_to_element_adj_table
            data["edge_to_edge_adj_table"]=self.edge_to_edge_adj_table
            data["edge_to_element_adj_table"]=self.edge_to_element_adj_table            
            data["element_to_element_adj_link"]=self.element_to_element_adj_link
            data["element_to_element_adj_table"]=self.element_to_element_adj_table
            data["element_to_edge_adj_table"]=self.element_to_edge_adj_table
            if 'polyhedron' in self.mesh_type:
                data["face"]=self.face
                data["face_to_element_adj_table"]=self.face_to_element_adj_table
                data["element_to_face_adj_table"]=self.element_to_face_adj_table
        torch.save(data,  filename)

    def load_from_torch(self, filename):
        if not os.path.isfile(filename):
            raise ValueError("not exist: "+filename)
        data=torch.load(filename, map_location="cpu")
        if "node" in data.keys():
            self.node=data["node"]
            if not isinstance(self.node, torch.Tensor):
                self.node=torch.tensor(self.node)
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
        if "mapping_from_node_name_to_index" in data.keys():
            self.mapping_from_node_name_to_index=data["mapping_from_node_name_to_index"]
        if "edge" in data.keys():
            self.edge=data["edge"]
        if "node_to_node_adj_link" in data.keys():
            self.node_to_node_adj_link=data["node_to_node_adj_link"]
        if "node_to_node_adj_table" in data.keys():
            self.node_to_node_adj_table=data["node_to_node_adj_table"]
        if "node_to_edge_adj_table" in data.keys():
            self.node_to_edge_adj_table=data["node_to_edge_adj_table"]
        if "node_to_element_adj_table" in data.keys():
            self.node_to_element_adj_table=data["node_to_element_adj_table"]
        if "edge_to_edge_adj_table" in data.keys():
            self.edge_to_edge_adj_table=data["edge_to_edge_adj_table"]
        if "edge_to_element_adj_table" in data.keys():
            self.edge_to_element_adj_table=data["edge_to_element_adj_table"]
        if "element_to_element_adj_link" in data.keys():
            self.element_to_element_adj_link=data["element_to_element_adj_link"]
        if "element_to_element_adj_table" in data.keys():
            self.element_to_element_adj_table=data["element_to_element_adj_table"]
        if "element_to_edge_adj_table" in data.keys():
            self.element_to_edge_adj_table=data["element_to_edge_adj_table"]
        if 'polyhedron' in self.mesh_type:
            if "face" in data.keys():
                self.face=data["face"]
            if "face_to_element_adj_table" in data.keys():
                self.face_to_element_adj_table=data["face_to_element_adj_table"]
            if "element_to_face_adj_table" in data.keys():
                self.element_to_face_adj_table=data["element_to_face_adj_table"]

    def copy(self, node, element, dtype=None, detach=True):
        if isinstance(node, torch.Tensor):
            if dtype is None:
                self.node=node.clone()
            else:
                self.node=node.clone().to(dtype)
            if detach==True:
                self.node=self.node.detach()
        elif isinstance(node, np.ndarray):
            if dtype is None:
                self.node=torch.tensor(node.copy())
            else:
                self.node=torch.tensor(node.copy(), dtype=dtype)
        elif isinstance(node, tuple) or isinstance(node, list):
            if dtype is None:
                self.node=torch.tensor(node, dtype=torch.float32)
            else:
                self.node=torch.tensor(node, dtype=dtype)
        else:
            raise NotImplementedError
        if isinstance(element, torch.Tensor):
            self.element=element.clone()
            if detach==True:
                self.element=self.element.detach()
        elif isinstance(element, np.ndarray):
            self.element=torch.tensor(element.copy(), dtype=torch.int64)
        elif isinstance(element, tuple) or isinstance(element, list):
            try:
                self.element=torch.tensor(element, dtype=torch.int64)
            except:
                self.element=deepcopy(element)
        else:
            raise NotImplementedError

    def copy_to_list(self, x):
        if isinstance(x, list):
            y=deepcopy(x)
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            y=x.tolist()
        elif isinstance(x, tuple) or isinstance(x, set):
            y=deepcopy(list(x))
        else:
            raise ValueError('unsupported python-object type')
        return y

    def make_element_copy(self, object_type=None):
        element=self.copy_to_list(self.element)
        if object_type is None:
            object_type=type(self.element)
        if object_type == 'list' or object_type == list:
            pass
        elif object_type == 'numpy' or object_type == np.ndarray:
            element=np.array(element, dtype=np.int64)
        elif object_type == 'torch' or object_type == torch.Tensor:
            element=torch.tensor(element, dtype=torch.int64)
        return element

    def make_node_copy(self, object_type=None, dtype=None):
        node=self.copy_to_list(self.node)
        if object_type is None:
            object_type=type(self.node)
        if dtype is None:
            dtype=self.node.dtype
        if object_type == 'list' or object_type == list:
            pass
        elif object_type == 'numpy' or object_type == np.ndarray:
            node=np.array(node, dtype=dtype)
        elif object_type == 'torch' or object_type == torch.Tensor:
            node=torch.tensor(node, dtype=dtype)
        return node
    
    def get_node_idx_by_name(self, name):
        if name in self.mapping_from_node_name_to_index.keys():
            return self.mapping_from_node_name_to_index[name]
        else:
            return None

    def set_node_name(self, idx, name, replace_old_name=False):
        if replace_old_name == True:
            self.mapping_from_node_name_to_index[name]=idx
        else:
            if name in self.mapping_from_node_name_to_index.keys():
                raise ValueError("node "+str(idx)+" already has a name:"+str(name))
            else:
                self.mapping_from_node_name_to_index[name]=idx
    
    def build_edge(self):
        #undirected edge represents a connection between two nodes
        #self.edge[k] is [node_idx_a, node_idx_b] and node_idx_a < node_idx_b: self.edge[k,0] < self.edge[k,1]
        #edge is determined by element_type
        #implement this function in a derived class (e.g. PolygonMesh)
        raise NotImplementedError

    def get_edge_idx_from_node_pair(self, nodeA_idx, nodeB_idx):
        # nodeA---an edge---nodeB
        if self.node_to_edge_adj_table is None:
            self.build_node_to_edge_adj_table()
        if nodeA_idx == nodeB_idx:
            return None
        edge_idx_listA=self.node_to_edge_adj_table[nodeA_idx]
        edge_idx_listB=self.node_to_edge_adj_table[nodeB_idx]
        temp=np.intersect1d(edge_idx_listA, edge_idx_listB, assume_unique=True)
        if len(temp) == 0:
            edge_idx=None
        elif len(temp) == 1:
            edge_idx=int(temp.item())
        else:
            raise ValueError("more than one edge between node "+str(nodeA_idx)+" and node "+str(nodeB_idx))
        return edge_idx

    def update_edge_length(self):
        if self.edge is None:
            self.build_edge()
        self.edge_length=Mesh.cal_edge_length(self.node, self.edge)

    @staticmethod
    def cal_edge_length(node, edge):
        x_j=node[edge[:,0]]
        x_i=node[edge[:,1]]
        edge_length=torch.norm(x_i-x_j, p=2, dim=1, keepdim=True)
        return edge_length

    def build_node_to_node_adj_link(self):
        #no self link
        #this is useful for GNN
        if self.edge is None:
            self.build_edge()
        adj_link=[]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if idx0 != idx1:
                adj_link.append([idx0, idx1])
                adj_link.append([idx1, idx0])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.node_to_node_adj_link=adj_link

    def build_node_to_node_adj_table(self):
        #no self link
        if self.edge is None:
            self.build_edge()
        adj_table=[[] for _ in range(self.node.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if idx0 != idx1:
                adj_table[idx0].append(idx1)
                adj_table[idx1].append(idx0)
        self.node_to_node_adj_table=adj_table

    def build_node_to_edge_adj_table(self):
        if self.edge is None:
            self.build_edge()
        adj_table=[[] for _ in range(self.node.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            adj_table[idx0].append(k)
            adj_table[idx1].append(k)
        self.node_to_edge_adj_table=adj_table

    def build_node_to_element_adj_table(self):
        #do not do this: node_to_element_adj_table=[[]]*self.node.shape[0]
        # a=[[]]*2=[[],[]], and a[0] and a[1] are the same object
        # a=[[],[]], and a[0] and a[1] are two different objects
        adj_table=[[] for _ in range(self.node.shape[0])]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(element)):
            e_m=element[m]
            for k in range(0, len(e_m)):
                adj_table[e_m[k]].append(m)
        self.node_to_element_adj_table=adj_table

    def build_edge_to_edge_adj_table(self):
        if self.node_to_edge_adj_table is None:
            self.build_node_to_edge_adj_table()
        adj_table=[[] for _ in range(self.edge.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            adj_table[k].extend(self.node_to_edge_adj_table[idx0])
            adj_table[k].extend(self.node_to_edge_adj_table[idx1])
            adj_table[k]=list(set(adj_table[k])-set([k]))
        self.edge_to_edge_adj_table=adj_table
        
    def build_edge_to_element_adj_table(self):
        if self.element_to_edge_adj_table is None:
            self.build_element_to_edge_adj_table()
        adj_table=[[] for _ in range(self.edge.shape[0])]
        for m in range(0, len(self.element)):
            adj_edge_idx_list=self.element_to_edge_adj_table[m]
            for edge_idx in adj_edge_idx_list:
                adj_table[edge_idx].append(m)
        self.edge_to_element_adj_table=adj_table

    def build_element_to_edge_adj_table(self):
        #this table is determined by element_type
        #implement this function in a derived class (e.g., PolygonMesh)
        raise NotImplementedError

    def build_element_to_element_adj_link_node(self):
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.node_to_element_adj_table)):
            e_set=self.node_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link["node"]=adj_link

    def build_element_to_element_adj_link_edge(self):
        if self.edge_to_element_adj_table is None:
            self.build_edge_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.edge_to_element_adj_table)):
            e_set=self.edge_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link["edge"]=adj_link

    def build_element_to_element_adj_link_face(self):
        if self.face_to_element_adj_table is None:
            self.build_face_to_element_adj_table()
        adj_link=[]
        for n in range(0, len(self.face_to_element_adj_table)):
            e_set=self.face_to_element_adj_table["face"][n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    adj_link.append([eid1, eid2])
                    adj_link.append([eid2, eid1])
        adj_link=torch.tensor(adj_link, dtype=torch.int64)
        adj_link=torch.unique(adj_link, dim=0, sorted=True)
        self.element_to_element_adj_link["face"]=adj_link

    def build_element_to_element_adj_link(self, adj):
        #no self link
        if adj not in ['node', 'edge', 'face']:
            raise ValueError('adj should be node, edge, or face')
        if adj == 'node':
            return self.build_element_to_element_adj_link_node()
        elif adj == 'edge':
            return self.build_element_to_element_adj_link_edge()
        elif adj == 'face':
            return self.build_element_to_element_adj_link_face()
        else:
            raise NotImplementedError

    def build_element_to_element_adj_table(self, adj):
        #no self link
        if self.element_to_element_adj_link[adj] is None:
            self.build_element_to_element_adj_link(adj=adj)
        adj_link=self.element_to_element_adj_link[adj]
        adj_table=[[] for _ in range(len(self.element))]
        for k in range(0, len(adj_link)):
            link=adj_link[k]
            idx0=int(link[0])
            idx1=int(link[1])
            if idx0 != idx1:
                adj_table[idx0].append(idx1)
        self.element_to_element_adj_table[adj]=adj_table

    def get_sub_mesh(self, element_idx_list, return_node_idx_list=False):
        #this function is slow: ony use it if the mesh has different types of elements
        new_element=[]
        node_idx_list=[]
        for m in range(0, len(element_idx_list)):
            elm=self.element[element_idx_list[m]]
            elm=self.copy_to_list(elm)
            new_element.append(elm)
            node_idx_list.extend(elm)
        node_idx_list=np.unique(node_idx_list)
        new_node=self.node[node_idx_list]
        #map old node idx to new node idx
        map={}
        for n in range(0, len(node_idx_list)):
            old_idx=node_idx_list[n]
            map[old_idx]=n #n is new_idx
        for m in range(0, len(new_element)):
            for n in range(0, len(new_element[m])):
                old_idx=new_element[m][n]
                new_element[m][n]=map[old_idx]
        new_element_type=None
        if self.element_type is not None:
            new_element_type=self.element_type[element_idx_list]
        new_mesh=Mesh(new_node, new_element, None, new_element_type, self.mesh_type)
        if return_node_idx_list == False:
            return new_mesh
        else:
            return new_mesh, node_idx_list
#%%
if __name__ == "__main__":
    #%%
    filename="D:/MLFEA/TAVR/FE/1908788_0_im_5_phase1_Root_solid_three_layers_aligned.vtk"
    root1=Mesh('polyhedron')
    root1.load_from_vtk(filename, dtype=torch.float32)
    root1.build_node_to_element_adj_table()
    #%%
    root1.save_by_vtk("test.vtk")
    #%%
    root2=Mesh('polyhedron')
    root2.load_from_vtk("test.vtk", dtype=torch.float32)
    #%%
    import time
    t1=time.time()
    root2.build_node_to_element_adj_table()
    t2=time.time()
    print('t2-t1', t2-t1)
    #%%
    t1=time.time()
    root2.build_element_adj_link(adj=1)
    t2=time.time()
    print('t2-t1', t2-t1)
