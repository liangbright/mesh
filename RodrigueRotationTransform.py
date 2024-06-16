import torch
import torch.nn as nn
from torch import cos, sin
from torch.linalg import matmul, norm
#%%  https://mathworld.wolfram.com/RodriguesRotationFormula.html
class RodrigueRotationTransform3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.axis=nn.Parameter(torch.rand(3))
        self.angle=nn.Parameter(torch.zeros(1))
        self.origin=nn.Parameter(torch.zeros(1,3))
        
    @staticmethod
    def cal_R(axis, angle):
        axis=axis/norm(axis, ord=2)
        a=axis[0]
        b=axis[1]
        c=axis[2]
        cos_t=cos(angle)
        sin_t=sin(angle)
        R=torch.zeros((3,3), dtype=a.dtype, device=a.device)
        R[0,0]=cos_t+a*a*(1-cos_t);    R[0,1]=a*b*(1-cos_t)-c*sin_t; R[0,2]=b*sin_t+a*c*(1-cos_t)
        R[1,0]=c*sin_t+a*b*(1-cos_t);  R[1,1]=cos_t+b*b*(1-cos_t);   R[1,2]=-a*sin_t+b*c*(1-cos_t)
        R[2,0]=-b*sin_t+a*c*(1-cos_t); R[2,1]=a*sin_t+b*c*(1-cos_t); R[2,2]=cos_t+c*c*(1-cos_t)
        return R
        
    def forward(self, x):
        #x.shape (N,3)
        #axis pass through self.origin
        R=RodrigueRotationTransform3D.cal_R(self.axis, self.angle)
        R=R.permute(1,0)#transpose
        y=matmul(x-self.origin, R)+self.origin
        return y
        