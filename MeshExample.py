import torch
from QuadMesh import Quad4Mesh as QuadMesh
from HexahedronMesh import Hex8Mesh as HexMesh
#%%
def create_quad_grid_mesh(Nx, Ny, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1), 4), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny, 3), dtype=dtype)
    map=torch.zeros((Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for y in range(0, Ny):
        for x in range(0, Nx):
            id+=1
            map[y,x]=id
            grid[id,0]=x
            grid[id,1]=y
            if y==0 or y==Ny-1 or x==0 or x==Nx-1:
                boundary.append(id)
    boundary=torch.tensor(boundary, dtype=torch.int64)
    id=-1
    for y in range(0, Ny-1):
        for x in range(0, Nx-1):
            id+=1
            element[id,0]=map[y,x]
            element[id,1]=map[y,x+1]
            element[id,2]=map[y+1,x+1]
            element[id,3]=map[y+1,x]
    grid_mesh=QuadMesh()
    grid_mesh.node=grid
    grid_mesh.element=element
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh
#%%
def create_hex_grid_mesh(Nx, Ny, Nz, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1)*(Nz-1), 8), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny*Nz, 3), dtype=dtype)
    map=torch.zeros((Nz, Ny, Nx), dtype=dtype)
    id=-1
    boundary=[]
    for z in range(0,Nz):
        for y in range(0, Ny):
            for x in range(0, Nx):
                id+=1
                map[z,y,x]=id
                grid[id,0]=x
                grid[id,1]=y
                grid[id,2]=z
                if y==0 or y==Ny-1 or x==0 or x==Nx-1 or z==0 or z==Nz-1:
                    boundary.append(id)
    boundary=torch.tensor(boundary, dtype=torch.int64)
    id=-1
    for z in range(0,Nz-1):
        for y in range(0, Ny-1):
            for x in range(0, Nx-1):
                id+=1
                element[id,0]=map[z,y,x]
                element[id,1]=map[z,y,x+1]
                element[id,2]=map[z,y+1,x+1]
                element[id,3]=map[z,y+1,x]
                element[id,4]=map[z+1,y,x]
                element[id,5]=map[z+1,y,x+1]
                element[id,6]=map[z+1,y+1,x+1]
                element[id,7]=map[z+1,y+1,x]
    grid_mesh=HexMesh()
    grid_mesh.node=grid
    grid_mesh.element=element
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh
#%%
if __name__ == '__main__':
    mesh0=create_quad_grid_mesh(10,20)
    mesh0.save_by_vtk("D:/MLFEA/TAA/mesh/quad_grid_mesh_x10y20.vtk")

    mesh1=create_hex_grid_mesh(10,20,2)
    mesh1.save_by_vtk("D:/MLFEA/TAA/mesh/hex_grid_mesh_x10y20z2.vtk")