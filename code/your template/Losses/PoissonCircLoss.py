import torch
import torch.nn as nn
import numpy as np
import math

class CircLoss():
    def __init__(self,n_points,int_pts,bound_pts,K):
        """
        n_points: number of points for domain
        int_pts = number of points per interface
        bound_ps = number of points per boundary
        K = [K_1,K_2]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.iter = 1
        self.criterion = nn.MSELoss()
        self.K=K
        
        self.n_points = n_points
        self.int_pts = int_pts
        self.bound_pts = bound_pts
        
        
        x = torch.rand(self.n_points)*2 - 1
        y = torch.rand(self.n_points)*2 - 1
        self.W = torch.stack((x, y), dim=1).reshape(-1,2)
        xx,yy = self.W[:,0],self.W[:,1]
        # 计算每个点到原点的距离
        #distances = torch.sqrt(xx**2 + yy**2)
        # 计算每个点的角度
        theta = torch.arctan2(self.W[:,1], self.W[:,0])
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)  # 将角度调整到[0, 2π]范围内
        # 计算每个点对应的花瓣半径
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.cos(theta)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.sin(theta)
        #r_petal_interp = torch.sqrt(x_petal**2 + y_petal**2)
        # 根据距离与花瓣半径的比较来确定点在内部还是外部
        mask1 = xx**2 + yy**2<x_petal**2 + y_petal**2
        mask2 = xx**2 + yy**2>x_petal**2 + y_petal**2
        # 选择内部区域的均匀采样点
        internal_points = self.W[mask1]
        if len(internal_points) > 300:
            internal_indices = torch.randperm(len(internal_points))[:300]
        else:
            internal_indices = torch.arange(len(internal_points))
        self.W1 = internal_points[internal_indices].to(self.device)
        # 选择外部区域的均匀采样点
        external_points = self.W[mask2]
        if len(external_points) > 500:
            external_indices = torch.randperm(len(external_points))[:500]
        else:
            external_indices = torch.arange(len(external_points))
        self.W2 = external_points[external_indices].to(self.device)
        
        angles = torch.linspace(0, 2 * torch.pi, self.int_pts) 
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.cos(angles)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.sin(angles)
        self.G = torch.stack((x_petal, y_petal), dim=1).reshape(-1,2).to(self.device)
        
        x = torch.arange(-1,1+(1/self.bound_pts),1/self.bound_pts)
        self.bx_1 = torch.stack((x,-torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.b1y = torch.stack((torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        self.bx1 = torch.stack((x,torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.b_1y = torch.stack((-torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        
        #All grads
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.G.requires_grad = True

    def reset(self):
        self.iter = 1
    
        x = torch.rand(self.n_points)*2 - 1
        y = torch.rand(self.n_points)*2 - 1
        self.W = torch.stack((x, y), dim=1).reshape(-1,2)
        xx,yy = self.W[:,0],self.W[:,1]
        # 计算每个点到原点的距离
        #distances = torch.sqrt(xx**2 + yy**2)
        # 计算每个点的角度
        theta = torch.arctan2(self.W[:,1], self.W[:,0])
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)  # 将角度调整到[0, 2π]范围内
        # 计算每个点对应的花瓣半径
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.cos(theta)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.sin(theta)
        #r_petal_interp = torch.sqrt(x_petal**2 + y_petal**2)
        # 根据距离与花瓣半径的比较来确定点在内部还是外部
        mask1 = xx**2 + yy**2<x_petal**2 + y_petal**2
        mask2 = xx**2 + yy**2>x_petal**2 + y_petal**2
        # 选择内部区域的均匀采样点
        internal_points = self.W[mask1]
        if len(internal_points) > 300:
            internal_indices = torch.randperm(len(internal_points))[:500]
        else:
            internal_indices = torch.arange(len(internal_points))
        self.W1 = internal_points[internal_indices].to(self.device)
        # 选择外部区域的均匀采样点
        external_points = self.W[mask2]
        if len(external_points) > 500:
            external_indices = torch.randperm(len(external_points))[:800]
        else:
            external_indices = torch.arange(len(external_points))
        self.W2 = external_points[external_indices].to(self.device)
        
        angles = torch.linspace(0, 2 * torch.pi, self.int_pts) 
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.cos(angles)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.sin(angles)
        self.G = torch.stack((x_petal, y_petal), dim=1).reshape(-1,2).to(self.device)
        
        x = torch.arange(-1,1+(1/self.bound_pts),1/self.bound_pts)
        self.bx_1 = torch.stack((x,-torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.b1y = torch.stack((torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        self.bx1 = torch.stack((x,torch.ones_like(x)),dim =1).reshape(-1,2).to(self.device)
        self.b_1y = torch.stack((-torch.ones_like(x),x),dim =1).reshape(-1,2).to(self.device)
        
        #All grads
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.G.requires_grad = True
            
    def loss(self,model,mode,cond_func=None):
        
        if(mode == 'ipinn'):
            model.apply_activation = lambda x : cond_func(x,'0')    
            u1 = model(self.W1)
            g1 = model(self.G)
            
            model.apply_activation = lambda x : cond_func(x,'1')    
            u2 = model(self.W2)
            g2 = model(self.G)
            ubx_1 = model(self.bx_1)
            ub1y = model(self.b1y)
            ubx1 = model(self.bx1)
            ub_1y = model(self.b_1y)
                
        elif(mode == 'xpinn'):
            u1 = model[0](self.W1)
            g1 = model[0](self.G)  
            
            u2 = model[1](self.W2)
            g2 = model[1](self.G)
            ubx_1 = model[1](self.bx_1)
            ub1y = model[1](self.b1y)
            ubx1 = model[1](self.bx1)
            ub_1y = model[1](self.b_1y)
            
        # Interface Grads
        dG1_dW = torch.autograd.grad(
                    outputs=g1,
                    inputs=self.G,
                    grad_outputs=torch.ones_like(g1),
                    retain_graph=True,
                    create_graph=True
                )[0] 
        dG2_dW = torch.autograd.grad(
                    outputs=g2,
                    inputs=self.G,
                    grad_outputs=torch.ones_like(g2),
                    retain_graph=True,
                    create_graph=True
                )[0]
        
        # Interface Grads
        x_interface = self.G[:, 0]
        y_interface = self.G[:, 1]

        # 确保 x_interface 和 y_interface 都是需要计算梯度的
        x_interface.requires_grad_(True)
        y_interface.requires_grad_(True)

        # 计算 u1 和 u2
        u1_interface = torch.sin(x_interface + y_interface) + torch.cos(x_interface + y_interface) + 1
        u2_interface = x_interface + y_interface + 1

        # 计算 u1 和 u2 对 x 和 y 的梯度
        grad_u1_x = torch.autograd.grad(outputs=u1_interface, inputs=x_interface, grad_outputs=torch.ones_like(u1_interface), retain_graph=True, create_graph=True)[0]
        grad_u1_y = torch.autograd.grad(outputs=u1_interface, inputs=y_interface, grad_outputs=torch.ones_like(u1_interface), retain_graph=True, create_graph=True)[0]
        grad_u2_x = torch.autograd.grad(outputs=u2_interface, inputs=x_interface, grad_outputs=torch.ones_like(u2_interface), retain_graph=True, create_graph=True)[0]
        grad_u2_y = torch.autograd.grad(outputs=u2_interface, inputs=y_interface, grad_outputs=torch.ones_like(u2_interface), retain_graph=True, create_graph=True)[0]

        # 合并梯度为张量
        grad_u1_tensor = torch.stack([grad_u1_x, grad_u1_y], dim=-1)
        grad_u2_tensor = torch.stack([grad_u2_x, grad_u2_y], dim=-1)

        # 计算 theta 和 petal
        theta = torch.atan2(y_interface, x_interface)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)

        x_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.cos(theta)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.sin(theta)
        r = x_petal**2 + y_petal**2

        # 计算 f
        f = x_interface**2 + y_interface**2 - r

        # 计算梯度 ∇f
        grad_f = torch.autograd.grad(
            outputs=f,
            inputs=self.G,
            grad_outputs=torch.ones_like(f),
            retain_graph=True,
            create_graph=True
        )[0]

        # 计算法向量 n2
        norm = torch.norm(grad_f, dim=1, keepdim=True)
        n2 = -grad_f / norm 
        # Calculate K \cdot (\nabla u)
        Ku1 = self.K[0](self.G[:, 0], self.G[:, 1]).unsqueeze(1) * dG1_dW  # 调整 K1 的输出形状
        Ku2 = self.K[1](self.G[:, 0], self.G[:, 1]).unsqueeze(1) * dG2_dW  # 调整 K2 的输出形状

        # Calculate (K \cdot (\nabla u)) \cdot n2
        result1 = torch.einsum('bi,bi->b', Ku1, n2)
        result2 = torch.einsum('bi,bi->b', Ku2, n2)
        # 定义 k1 和 k2
        K1_interface = (x_interface**2 - y_interface**2 + 3) / 7
        K2_interface = (2 + x_interface * y_interface) / 5

        # 计算 Ku1 和 Ku2
        Ku1_interface = K1_interface.unsqueeze(1) * grad_u1_tensor
        Ku2_interface = K2_interface.unsqueeze(1) * grad_u2_tensor

        # 计算 (Ku1) · n2 和 (Ku2) · n2
        result1_interface = torch.einsum('bi,bi->b', Ku1_interface, n2)
        result2_interface = torch.einsum('bi,bi->b', Ku2_interface, n2)

        # 计算 phi2
        phi2 = result2_interface - result1_interface
        loss_deriv = self.criterion(result2 - result1, phi2)
        phi1 = (x_interface + y_interface + 1) - (torch.sin(x_interface + y_interface) + torch.cos(x_interface + y_interface) + 1)
        loss_jump = self.criterion(g2 - g1, phi1)

        # 总损失
        loss_interface = loss_jump + loss_deriv
        # PDE grads
        du_dW1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=u1,
            grad_outputs=torch.ones_like(u1),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dxx1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=du_dW1[:,0],
            grad_outputs=torch.ones_like(du_dW1[:,0]),
            retain_graph=True,
            create_graph=True
        )[0][:,0]
        du_dyy1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=du_dW1[:,1],
            grad_outputs=torch.ones_like(du_dW1[:,1]),
            retain_graph=True,
            create_graph=True
        )[0][:,1]
        
        du_dW2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=u2,
            grad_outputs=torch.ones_like(u2),
            retain_graph=True,
            create_graph=True
        )[0]
        du_dxx2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=du_dW2[:,0],
            grad_outputs=torch.ones_like(du_dW2[:,0]),
            retain_graph=True,
            create_graph=True
        )[0][:,0]
        du_dyy2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=du_dW2[:,1],
            grad_outputs=torch.ones_like(du_dW2[:,1]),
            retain_graph=True,
            create_graph=True
        )[0][:,1]
        
         
        K1 = (self.W1[:,0]**2 - self.W1[:,1]**2 + 3) / 7
        K1_grad_u1_x = K1 * du_dW1[:,0]

        K1_grad_u1_y = K1 * du_dW1[:, 1]

        grad_K1_grad_u1_x = torch.autograd.grad(
                outputs=K1_grad_u1_x,
                inputs=self.W1,
                grad_outputs=torch.ones_like(K1_grad_u1_x),
                retain_graph=True,
                create_graph=True
            )[0][:, 0]

        grad_K1_grad_u1_y = torch.autograd.grad(
                outputs=K1_grad_u1_y,
                inputs=self.W1,
                grad_outputs=torch.ones_like(K1_grad_u1_y),
                retain_graph=True,
                create_graph=True
            )[0][:, 1]
        grad_K1_grad_u1=grad_K1_grad_u1_x + grad_K1_grad_u1_y
        f1 = -(2*(self.W1[:,0]**2 - self.W1[:,1]**2 + 3) /7 * (torch.cos(self.W1[:,0]+ self.W1[:,1])-torch.sin(self.W1[:,0]+self.W1[:,1]))+((2*(self.W1[:,0]-self.W1[:,1])/7)*(torch.cos(self.W1[:,0]+ self.W1[:,1])-torch.sin(self.W1[:,0]+self.W1[:,1]))))
                

        loss_pde1= self.criterion(-grad_K1_grad_u1,f1) 

        K2 = (2 + self.W2[:,0] * self.W2[:,1]) / 5

    # 计算 K2乘以梯度
        K2_grad_u2_x = K2 * du_dW2[:,0]
        K2_grad_u2_y = K2 * du_dW2[:, 1]
        # 计算 ∇(K2∇u2)
        grad_K2_grad_u2_x = torch.autograd.grad(
            outputs=K2_grad_u2_x,
            inputs=self.W2,
            grad_outputs=torch.ones_like(K2_grad_u2_x),
            retain_graph=True,
            create_graph=True
        )[0][:, 0]

        grad_K2_grad_u2_y = torch.autograd.grad(
            outputs=K2_grad_u2_y,
            inputs=self.W2,
            grad_outputs=torch.ones_like(K2_grad_u2_y),
            retain_graph=True,
            create_graph=True
        )[0][:, 1]
        grad_K2_grad_u2=grad_K2_grad_u2_x + grad_K2_grad_u2_y
        f2=-(self.W2[:,1] / 5 + self.W2[:,0] / 5)
        loss_pde2=self.criterion(-grad_K2_grad_u2,f2)

        loss_boundary = (self.criterion(ubx_1, (self.bx_1[:, 0]).view_as(ubx_1)) + 
                         self.criterion(ub_1y, (self.b_1y[:,1]).view_as(ub_1y)) + 
                         self.criterion(ubx1, (self.bx1[:, 0]).view_as(ubx1) + torch.full_like(ubx1, 2)) + 
                         self.criterion(ub1y, (self.b1y[:, 1]).view_as(ub1y) + torch.full_like(ubx1, 2)))
                                  
        w1=0.00001
        w2=0.00001
        w3=0.99999
        w4=0.99999
        loss=w1*loss_pde1+w3*loss_interface+w2*loss_pde2+w4*loss_boundary
        loss.backward()
        #print('PDE: ',loss_pde.item(),'Bound: ',loss_boundary.item(),'jmp: ',loss_jump.item(),'deriv: ',loss_deriv.item())
        return loss     