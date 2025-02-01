import torch
import numpy as np
from PolygonMeshProcessing import QuadMesh
from HexahedronMesh import HexahedronMesh as HexMesh
#%%
def create_quad_cylinder_mesh(n_rings, n_points_per_ring, dtype=torch.float32):
    theta=2*np.pi/n_points_per_ring
    node=torch.zeros((n_rings*n_points_per_ring, 3), dtype=dtype)
    k=-1
    for n in range(0, n_rings):
        for m in range(0, n_points_per_ring):
            x=np.cos(theta*m)
            y=np.sin(theta*m)
            z=n/n_rings
            k=k+1
            node[k,0]=x
            node[k,1]=y
            node[k,2]=z
    element=[]
    for n in range(1, n_rings):
        idxA=np.arange((n-1)*n_points_per_ring, n*n_points_per_ring)
        idxB=np.arange(n*n_points_per_ring, (n+1)*n_points_per_ring)
        for i in range(0, n_points_per_ring-1):
            element.append([idxA[i], idxA[i+1], idxB[i+1], idxB[i]])
        element.append([idxA[n_points_per_ring-1], idxA[0], idxB[0], idxB[n_points_per_ring-1]])
    cylinder=QuadMesh(node, element)
    return cylinder
#%%
def create_quad_grid_mesh(Nx, Ny, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1), 4), dtype=torch.int64)
    node=torch.zeros((Nx*Ny, 3), dtype=dtype)
    map=torch.zeros((Ny, Nx), dtype=torch.int64)
    id=-1
    boundary=[]
    for y in range(0, Ny):
        for x in range(0, Nx):
            id+=1
            map[y,x]=id
            node[id,0]=x
            node[id,1]=y
            if y==0 or y==Ny-1 or x==0 or x==Nx-1:
                boundary.append(id)
    #-----------------------------------
    id=-1
    for y in range(0, Ny-1):
        for x in range(0, Nx-1):
            id+=1
            element[id,0]=map[y,x]
            element[id,1]=map[y,x+1]
            element[id,2]=map[y+1,x+1]
            element[id,3]=map[y+1,x]
    grid_mesh=QuadMesh(node, element)
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh
#%%
def create_hex_grid_mesh(Nx, Ny, Nz, dtype=torch.float32):
    element=torch.zeros(((Nx-1)*(Ny-1)*(Nz-1), 8), dtype=torch.int64)
    grid=torch.zeros((Nx*Ny*Nz, 3), dtype=dtype)
    map=torch.zeros((Nz, Ny, Nx), dtype=torch.int64)
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
                if z==0 or z==Nz-1 or y==0 or y==Ny-1 or x==0 or x==Nx-1:
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
    grid_mesh=HexMesh(grid, element)
    grid_mesh.node_set['boundary']=boundary
    return grid_mesh
#%%
def create_quad_mesh_square_in_square_simple(n_squares=2):
#0-----7-----6
#|     _     |
#1    |_|    5
#|           |
#2-----3-----4
    if n_squares < 2:
        raise ValueError("n_squares must be >= 2")
    square0=torch.tensor([[-1,1,0], [-1,0,0], [-1,-1,0], [0,-1,0], [1,-1,0], [1,0,0], [1,1,0], [0,1,0]],
                         dtype=torch.float32)
    squareN=square0/n_squares
    square_list=[square0]
    for n in range(1, n_squares-1):
        square=square0+(squareN-square0)*(n/(n_squares-1))
        square_list.append(square)
    square_list.append(squareN)
    element=[]; node=[]
    K=len(square0)
    for n in range(0, n_squares-1):
        square_n=square_list[n]
        #square_n1=square_list[n+1]
        node.extend(square_n.tolist())        
        for m in range(0, K):
            if m < K-1:
                idx0=n*K+m
                idx1=n*K+m+1
                idx2=(n+1)*K+m+1
                idx3=(n+1)*K+m
            else:
                idx0=n*K+m
                idx1=n*K
                idx2=(n+1)*K
                idx3=(n+1)*K+m
            element.append([idx0, idx1, idx2, idx3])
    node.extend(squareN.tolist())
    #seal the hole    
    node.append([0,0,0])#center node
    N=n_squares-1
    element.append([(N+1)*K, N*K+7, N*K+0, N*K+1])
    element.append([(N+1)*K, N*K+1, N*K+2, N*K+3])
    element.append([(N+1)*K, N*K+3, N*K+4, N*K+5])
    element.append([(N+1)*K, N*K+5, N*K+6, N*K+7])
    square_mesh=QuadMesh(node, element)
    return square_mesh        
#%%
def create_quad_mesh_square_in_square(n_squares=2, Nx=3, Ny=3):
    if n_squares < 2:
        raise ValueError("n_squares must be >= 2")
    inner_mesh=create_quad_grid_mesh(Nx, Ny)
    inner_mesh.node-=inner_mesh.node.mean(dim=0, keepdim=True)
    inner_mesh.node/=max(Nx, Ny)*n_squares    
    #D-----------C
    #|     _     |
    #|    |_|    |
    #|           |
    #A-----------B
    square0_idx=(np.arange(0,Nx).tolist()+ [Nx-1+Nx*n for n in range(1, Ny)] + [Nx*Ny-1-n for n in range(1, Nx)]
                 + [Nx*(Ny-1)-Nx*n for n in range(1,Ny-1)])
    square0=inner_mesh.node[square0_idx]        
    element=inner_mesh.element.tolist()
    node=inner_mesh.node.tolist()
    idx_start=len(node)
    node.extend(square0.tolist())
    K=len(square0)
    squareN=square0*n_squares
    for n in range(1, n_squares):
        square_n=square0+(squareN-square0)*n/(n_squares-1) 
        node.extend(square_n.tolist())        
        for m in range(0, K):
            if m < K-1:
                idx0=idx_start+n*K+m
                idx1=idx_start+n*K+m+1
                idx2=idx_start+(n-1)*K+m+1
                idx3=idx_start+(n-1)*K+m
            else:
                idx0=idx_start+n*K+m
                idx1=idx_start+n*K
                idx2=idx_start+(n-1)*K
                idx3=idx_start+(n-1)*K+m
            element.append([idx0, idx1, idx2, idx3])
    square_mesh=QuadMesh(node, element)
    return square_mesh
#%%
if __name__ == '__main__':
    mesh0=create_quad_grid_mesh(10,20)
    mesh0.save_as_vtk("D:/MLFEA/TAA/mesh/quad_grid_mesh_x10y20.vtk")

    mesh1=create_hex_grid_mesh(10,20,2)
    mesh1.save_as_vtk("D:/MLFEA/TAA/mesh/hex_grid_mesh_x10y20z2.vtk")
    #%%
    mesh2=create_quad_mesh_square_in_square(n_squares=3, Nx=3, Ny=3)
    mesh2.save_as_vtk("D:/MLFEA/TAA/mesh/quad_mesh_square_in_square.vtk")
    #mesh2=mesh2.subdivide_to_quad()
    #mesh2=mesh2.subdivide_to_quad()
    #mesh2.save_as_vtk("D:/MLFEA/TAA/mesh/quad_mesh_square_in_square_sub2.vtk")