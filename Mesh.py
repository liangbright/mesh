# -*- coding: utf-8 -*-
"""
Created on Mon May 16 22:07:37 2022

@author: liang
"""
import torch
import numpy as np
from copy import deepcopy
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
        self.element_type=None #None then element_type will be infered by using mesh_type and node count per element
        self.node_name_to_index={} #e.g., {'landmark1':10}
        self.node_set={} #e.g., {'set1':[0,1,2]}
        self.element_set={} #e.g., {'set1':[1,3,5]}
        self.node_data={} #e.g., {'stress':stress}, stress is Nx9 2D array
        self.element_data={} #e.g., {'stress':stress}, stress is Mx9 2D array
        self.mesh_data={} # it is only saved by torch
        self.edge=None #only one edge between two adj nodes
        self.node_adj_link=None
        self.node_adj_table=None
        self.element_adj_link={} #{"adj1":None}
        self.element_adj_table={} #{"adj1":None}
        self.node_to_element_adj_table=None
        self.node_to_edge_adj_table=None
        self.edge_to_element_adj_table={"adj1":None, "adj2":None}

    def load_from_vtk(self, filename, dtype):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot load from vtk")
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
            elif n_nodes == 10:
                cell_type=vtk.VTK_QUADRATIC_TETRA
            else:
                cell_type=vtk.VTK_POLYHEDRON
        elif 'polygon' in element_type:
            if n_nodes == 3:
                cell_type=vtk.VTK_TRIANGLE
            elif n_nodes == 4:
                cell_type=vtk.VTK_QUAD
            elif n_nodes == 6:
                cell_type=vtk.VTK_QUADRATIC_TRIANGLE
            else:
                cell_type=vtk.VTK_POLYGON
        else:
            raise ValueError('unknown element_type:'+element_type)
        return cell_type

    def convert_to_vtk(self):
        if _Flag_VTK_IMPORT_ == False:
            print("cannot convert to vtk")
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

    def save_by_vtk(self, filename, ascii=True, vtk42=True, use_vtk=True):
        if _Flag_VTK_IMPORT_ == False or use_vtk == False:
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

    def save_by_torch(self, filename, save_link=False):
        data={"mesh_type":self.mesh_type,
              "node":self.node,
              "element":self.element,
              "node_set":self.node_set,
              "element_set":self.element_set,
              "node_data":self.node_data,
              "element_data":self.element_data,
              "mesh_data":self.mesh_data,
              "node_name_to_index":self.node_name_to_index}
        if save_link == True:
            data["edge"]=self.edge,
            data["node_adj_link"]=self.node_adj_link,
            data["element_adj_link"]=self.element_adj_link,
            data["node_to_element_adj_table"]=self.node_to_element_adj_table,
            data["node_to_edge_adj_table"]=self.node_to_edge_adj_table,
            data["edge_to_element_adj_table"]=self.edge_to_element_adj_table
        torch.save(data,  filename)

    def load_from_torch(self, filename):
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
        if "node_name_to_index" in data.keys():
            self.node_name_to_index=data["node_name_to_index"]
        if "node_to_element_adj_table" in data.keys():
            self.node_to_element_adj_table=data["node_to_element_adj_table"]
        if "edge" in data.keys():
            self.edge=data["edge"]
        if "node_adj_link" in data.keys():
            self.node_adj_link=data["node_adj_link"]
        if "element_adj_link" in data.keys():
            self.element_adj_link=data["element_adj_link"]

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
            m_list=[]
            for m in range(0, len(element)):
                m_list.append(len(element[m]))
            if max(m_list) == min(m_list):
                self.element=torch.tensor(element, dtype=torch.int64)
            else:
                self.element=deepcopy(element)
        else:
            raise NotImplementedError

    def build_edge(self):
        raise NotImplementedError

    def build_node_adj_link(self):
        raise NotImplementedError

    def build_node_adj_table(self):
        #no self link
        if self.edge is None:
            self.build_edge()
        node_adj_table=[[] for _ in range(self.node.shape[0])]
        for k in range(0, len(self.edge)):
            idx0=int(self.edge[k,0])
            idx1=int(self.edge[k,1])
            if idx0 != idx1:
                node_adj_table[idx0].append(idx1)
                node_adj_table[idx1].append(idx0)
        self.node_adj_table=node_adj_table

    def build_node_to_element_adj_table(self):
        #do not do this: node_to_element_adj_table=[[]]*self.node.shape[0]
        # a=[[]]*2=[[],[]], and a[0] and a[1] are the same object
        # a=[[],[]], and a[0] and a[1] are two different objects
        node_to_element_adj_table=[[] for _ in range(self.node.shape[0])]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for m in range(0, len(element)):
            e_m=element[m]
            for k in range(0, len(e_m)):
                node_to_element_adj_table[e_m[k]].append(m)
        self.node_to_element_adj_table=node_to_element_adj_table

    def build_element_adj_link_adj1(self):
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        element_adj_link=[]
        for n in range(0, len(self.node_to_element_adj_table)):
            e_set=self.node_to_element_adj_table[n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    element_adj_link.append([eid1, eid2])
                    element_adj_link.append([eid2, eid1])
        element_adj_link=torch.tensor(element_adj_link, dtype=torch.int64)
        element_adj_link=torch.unique(element_adj_link, dim=0, sorted=True)
        self.element_adj_link["adj1"]=element_adj_link

    def build_element_adj_link_adj2(self):
        if self.edge_to_element_adj_table["adj2"] is None:
            self.build_edge_to_element_adj_table(adj=2)
        element_adj_link=[]
        for n in range(0, len(self.edge_to_element_adj_table)):
            e_set=self.edge_to_element_adj_table["adj2"][n]
            for m1 in range(0, len(e_set)):
                for m2 in range(m1+1, len(e_set)):
                    eid1=e_set[m1]; eid2=e_set[m2]
                    element_adj_link.append([eid1, eid2])
                    element_adj_link.append([eid2, eid1])
        element_adj_link=torch.tensor(element_adj_link, dtype=torch.int64)
        element_adj_link=torch.unique(element_adj_link, dim=0, sorted=True)
        self.element_adj_link["adj2"]=element_adj_link

    def build_element_adj_link(self, adj):
        #two elements are adj if they share at least adj node(s)
        #no self link
        if self.element_adj_link is None:
            self.element_adj_link={}
        if adj < 1:
            raise ValueError('adj should be >=1, adj='+str(adj))
        if adj == 1:
            #two elements are adj if they share at least one(adj) node
            return self.build_element_adj_link_adj1()
        elif adj == 2:
            #two elements are adj if they share at least two(adj) nodes
            return self.build_element_adj_link_adj2()
        #---------the code is very slow -------------------
        element_adj_link=[]
        element=self.element
        if isinstance(element, torch.Tensor):
            element=element.detach().cpu().numpy()
        for n in range(0, len(element)):
            e_n=element[n]
            for m in range(n+1, len(element)):
                e_m=element[m]
                temp=np.isin(e_n, e_m, assume_unique=True)
                temp=np.sum(temp)
                if temp >= adj:
                    element_adj_link.append([n, m])
                    element_adj_link.append([m, n])
        element_adj_link=torch.tensor(element_adj_link, dtype=torch.int64)
        element_adj_link=torch.unique(element_adj_link, dim=0, sorted=True)
        self.element_adj_link["adj"+str(adj)]=element_adj_link

    def build_element_adj_table(self, adj):
        #no self link
        try:
            element_adj_link=self.element_adj_link["adj"+str(adj)]
        except:
            self.build_element_adj_link(adj=adj)
            element_adj_link=self.element_adj_link["adj"+str(adj)]
        if element_adj_link is None:
            self.build_element_adj_link(adj=adj)
            element_adj_link=self.element_adj_link["adj"+str(adj)]
        element_adj_table=[[] for _ in range(len(self.element))]
        for k in range(0, len(element_adj_link)):
            link=element_adj_link[k]
            idx0=int(link[0])
            idx1=int(link[1])
            if idx0 != idx1:
                element_adj_table[idx0].append(idx1)
        if self.element_adj_table is None:
            self.element_adj_table={}
        self.element_adj_table["adj"+str(adj)]=element_adj_table

    def build_node_to_edge_adj_table(self):
        if self.edge is None:
            self.build_edge()
        node_to_edge_adj_table=[[] for _ in range(self.edge.shape[0])]
        for k in range(0, len(self.edge)):
            node_to_edge_adj_table[self.edge[k,0]].append(k)
            node_to_edge_adj_table[self.edge[k,1]].append(k)
        self.node_to_edge_adj_table=node_to_edge_adj_table

    def build_edge_to_element_adj_table(self, adj):
        #an edge is adj to an element if the one (adj=1) or two(adj=2) nodes of the egde belong to the element
        #if adj=1, then Polygon.find_boundary_node will fail
        if adj < 1 or adj > 2:
            raise ValueError('adj can only be 1 or 2, adj='+str(adj))
        if self.edge is None:
            self.build_edge()
        if self.node_to_element_adj_table is None:
            self.build_node_to_element_adj_table()
        edge_to_element_adj_table=[]
        for k in range(0, len(self.edge)):
            elm_set0=self.node_to_element_adj_table[self.edge[k,0]]
            elm_set1=self.node_to_element_adj_table[self.edge[k,1]]
            if adj == 1:
                elm_set=list(set(list(elm_set0)+list(elm_set1)))
            elif adj == 2:
                elm_set=list(np.intersect1d(elm_set0, elm_set1))
            else:
                raise ValueError("not possible")
            edge_to_element_adj_table.append(elm_set)
        if self.edge_to_element_adj_table is None:
            self.edge_to_element_adj_table={"adj1":None, "adj2":None}
        if adj == 1:
            self.edge_to_element_adj_table["adj1"]=edge_to_element_adj_table
        if adj == 2:
            self.edge_to_element_adj_table["adj2"]=edge_to_element_adj_table

    def _elm_to_list(self, elm):
        #elm is self.element[m]
        if isinstance(elm, list):
            pass
        elif isinstance(elm, torch.Tensor):
            elm=elm.cpu().numpy().tolist()
        elif isinstance(elm, np.ndarray):
            elm=elm.tolist()
        elif isinstance(elm, tuple):
            elm=list(elm)
        else:
            raise ValueError('unsupported type')
        return elm

    def get_sub_mesh(self, element_idx_list):
        #this function is slow: ony use it if the mesh has different types of elements
        new_element=[]
        used_old_idx_list=[]
        for m in range(0, len(element_idx_list)):
            elm=self.element[element_idx_list[m]]
            elm=self._elm_to_list(elm)
            new_element.append(elm)
            used_old_idx_list.extend(elm)
        used_old_idx_list=np.unique(used_old_idx_list)
        new_node=self.node[used_old_idx_list]
        #map old node idx to new node idx
        map={}
        for n in range(0, len(used_old_idx_list)):
            old_idx=used_old_idx_list[n]
            map[old_idx]=n #n is new_idx
        for m in range(0, len(new_element)):
            for n in range(0, len(new_element[m])):
                old_idx=new_element[m][n]
                new_element[m][n]=map[old_idx]
        new_mesh=Mesh(self.mesh_type)
        new_mesh.node=new_node
        new_mesh.element=new_element
        return new_mesh
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
