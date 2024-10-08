import numpy as np
import torch
import math

from Template_PDE import AbstractPDE

class Poisson2DST(AbstractPDE):
    """
    The Exact solution for Two dimensional Poisson Equation with dirchlet boundary conditions
    ∇(K_m ∇u) = 1 in Ω_m
    [[u]] = 0 in Γ_int
    [[K∇u]].n_2 = -y/sqrt(2) in Γ_int
    u_m = (Λ_m)^d    
    """
    def __init__(self):
        super().__init__()

    def solution(self,x, y):
        if(y <= x):
            return x**2 + x*y
        else:
            return x**2 + y**2 
        
class Poisson2DCirc(AbstractPDE):
    def __init__(self):
        pass  # 不需要 np.vectorize，因为我们将直接处理张量

    def solution(self, x, y):
        # 确保 x 和 y 是 torch.Tensor 类型
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float()

        # 计算角度 theta
        theta = torch.arctan2(y, x)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

        # 计算花瓣半径
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.cos(theta)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.sin(theta)

        # 计算点是否在花瓣内
        inside_mask = (x**2 + y**2) < (x_petal**2 + y_petal**2)

        # 使用 torch.where 进行条件判断
        result = torch.where(inside_mask, torch.sin(x + y) + torch.cos(x + y) + 1, x + y + 1)
        
        return result          