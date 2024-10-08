import torch
import torch.nn as nn
import numpy as np
import math

class CircLoss():
    def __init__(self, n_points, int_pts, bound_pts, internal_points, external_points, K):
        """
        n_points: number of points for domain
        int_pts: number of points per interface
        bound_pts: number of points per boundary
        internal_points: number of points for internal domain
        external_points: number of points for external domain
        K = [K_1, K_2]
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.MSELoss()
        self.K = K

        self.n_points = n_points
        self.int_pts = int_pts
        self.bound_pts = bound_pts
        self.internal_points = internal_points
        self.external_points = external_points
        
    
        # Initialize internal and external points
        x = torch.rand(self.n_points) * 2 - 1
        y = torch.rand(self.n_points) * 2 - 1
        self.W = torch.stack((x, y), dim=1).reshape(-1, 2)
        xx, yy = self.W[:, 0], self.W[:, 1]
        theta = torch.arctan2(yy, xx)
        theta = torch.where(theta < 0, theta + 2 * torch.pi, theta)
        
        # Define petal shape (example of domain)
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.cos(theta)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * theta) * torch.sin(6 * theta)) * torch.sin(theta)

        mask1 = xx**2 + yy**2 < x_petal**2 + y_petal**2
        mask2 = xx**2 + yy**2 > x_petal**2 + y_petal**2

        points1 = self.W[mask1]
        points2 = self.W[mask2]

        # Boundary points (Rectangular domain boundary)
        x = torch.arange(-1, 1 + (1 / self.bound_pts), 1 / self.bound_pts)
        self.bx_1 = torch.stack((x, -torch.ones_like(x)), dim=1).reshape(-1, 2).to(self.device)
        self.b1y = torch.stack((torch.ones_like(x), x), dim=1).reshape(-1, 2).to(self.device)
        self.bx1 = torch.stack((x, torch.ones_like(x)), dim=1).reshape(-1, 2).to(self.device)
        self.b_1y = torch.stack((-torch.ones_like(x), x), dim=1).reshape(-1, 2).to(self.device)

        
        """
        初始化界面点，基于 int_pts 调整分割的数量
        """
        angles = torch.linspace(0, 2 * torch.pi, self.int_pts)
        x_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.cos(angles)
        y_petal = (0.40178 + 0.40178 * torch.cos(2 * angles) * torch.sin(6 * angles)) * torch.sin(angles)
        self.G = torch.stack((x_petal, y_petal), dim=1).reshape(-1, 2).to(self.device)

        if len(points1) > internal_points:
            internal_indices = torch.randperm(len(points1))[:internal_points]
        else:
            internal_indices = torch.arange(len(points1))
        self.W1 = points1[internal_indices].to(self.device)

        if len(points2) > self.external_points:
            external_indices = torch.randperm(len(points2))[:external_points]
        else:
            external_indices = torch.arange(len(external_points))
        self.W2 = points2[external_indices].to(self.device)

        # Enable gradient computation
        self.W1.requires_grad = True
        self.W2.requires_grad = True
        self.G.requires_grad = True


            
    def loss(self,model,cond_func=None):
        
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
     
                    create_graph=True
                )[0] 
        dG2_dW = torch.autograd.grad(
                    outputs=g2,
                    inputs=self.G,
                    grad_outputs=torch.ones_like(g2),
            
                    create_graph=True
                )[0]
        

        
        # Interface Grads
        x_interface = self.G[:, 0]
        y_interface = self.G[:, 1]

        # 确保 x_interface 和 y_interface 都是需要计算梯度的
        x_interface.requires_grad_(True)
        y_interface.requires_grad_(True)


        u1_interface = torch.sin(x_interface + y_interface) + torch.cos(x_interface + y_interface) + 1
        u2_interface = x_interface + y_interface + 1

        # 计算 u1 和 u2 对 x 和 y 的梯度
        grad_u1_x = torch.autograd.grad(outputs=u1_interface, inputs=x_interface, grad_outputs=torch.ones_like(u1_interface), create_graph=True)[0]
        grad_u1_y = torch.autograd.grad(outputs=u1_interface, inputs=y_interface, grad_outputs=torch.ones_like(u1_interface), create_graph=True)[0]
        grad_u2_x = torch.autograd.grad(outputs=u2_interface, inputs=x_interface, grad_outputs=torch.ones_like(u2_interface),  create_graph=True)[0]
        grad_u2_y = torch.autograd.grad(outputs=u2_interface, inputs=y_interface, grad_outputs=torch.ones_like(u2_interface), create_graph=True)[0]
        

        # 合并梯度为张量
        grad_u1_tensor = torch.stack([grad_u1_x.detach(), grad_u1_y.detach()], dim=-1)
        grad_u2_tensor = torch.stack([grad_u2_x.detach(), grad_u2_y.detach()], dim=-1)

        

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
        phi1 = (x_interface + y_interface + 1) - (torch.sin(x_interface + y_interface) + torch.cos(x_interface + y_interface) + 1)
        phi1 = phi1.unsqueeze(1)  # 使 phi1 变为 [n, 1] 形状
        loss_deriv = self.criterion(result2 - result1, phi2)
        

        loss_jump = self.criterion(g2 - g1, phi1)


        # 总损失
        loss_interface = loss_jump + loss_deriv
        # PDE grads
        du_dW1 = torch.autograd.grad(
            inputs=self.W1,
            outputs=u1,
            grad_outputs=torch.ones_like(u1),
            #reate_graph=True
        )[0]

        
        du_dW2 = torch.autograd.grad(
            inputs=self.W2,
            outputs=u2,
            grad_outputs=torch.ones_like(u2),
            #create_graph=True
        )[0]
        
         
        K1 = (self.W1[:,0]**2 - self.W1[:,1]**2 + 3) / 7
        K1_grad_u1_x = (K1) * du_dW1[:,0]

        K1_grad_u1_y = (K1) * du_dW1[:, 1]

        grad_K1_grad_u1_x = torch.autograd.grad(
                outputs=K1_grad_u1_x,
                inputs=self.W1,
                grad_outputs=torch.ones_like(K1_grad_u1_x),
                #retain_graph=True,
                create_graph=True
            )[0][:, 0]

        grad_K1_grad_u1_y = torch.autograd.grad(
                outputs=K1_grad_u1_y,
                inputs=self.W1,
                grad_outputs=torch.ones_like(K1_grad_u1_y),
                #retain_graph=True,
                create_graph=True
            )[0][:, 1]
        grad_K1_grad_u1=grad_K1_grad_u1_x + grad_K1_grad_u1_y
        
        self.W1 = self.W1.clone().detach().requires_grad_(True)

        # 定义 u1 和 K1
        u1_r = torch.sin(self.W1[:,0] + self.W1[:,1]) + torch.cos(self.W1[:,0] + self.W1[:,1]) + 1
        K1_r = (self.W1[:,0]**2 - self.W1[:,1]**2 + 3) / 7

        # 计算 u1 的梯度 (∇u1)
        grad_u1 = torch.autograd.grad(u1_r.sum(), self.W1, create_graph=True)[0]  # 求和确保输出为标量
        grad_u1_x_r = grad_u1[:, 0]  # u1 对 x 的偏导
        grad_u1_y_r = grad_u1[:, 1]  # u1 对 y 的偏导

        # 将梯度与 K1 相乘 (K1 * ∇u1)
        K1_grad_u1_x_r = K1_r * grad_u1_x_r
        K1_grad_u1_y_r = K1_r * grad_u1_y_r

        # 计算散度 ∇·(K1∇u1)
        # 需要考虑 K1 对 x 和 y 的梯度贡献
        grad_K1_u1_x = torch.autograd.grad(K1_grad_u1_x_r.sum(), self.W1, create_graph=True)[0][:, 0]
        grad_K1_u1_y = torch.autograd.grad(K1_grad_u1_y_r.sum(), self.W1, create_graph=True)[0][:, 1]

        # 计算散度，并取负号
        f1 = -(grad_K1_u1_x + grad_K1_u1_y)

        # 计算损失
        loss_pde1= self.criterion(-(grad_K1_grad_u1),f1)  # 这里假设目标是 0，具体视损失函数而定

                

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
            #retain_graph=True,
            create_graph=True
        )[0][:, 0]

        grad_K2_grad_u2_y = torch.autograd.grad(
            outputs=K2_grad_u2_y,
            inputs=self.W2,
            grad_outputs=torch.ones_like(K2_grad_u2_y),
            #retain_graph=True,
            create_graph=True
        )[0][:, 1]
        grad_K2_grad_u2=grad_K2_grad_u2_x + grad_K2_grad_u2_y
        f2=-(self.W2[:,1] / 5 + self.W2[:,0] / 5)
        loss_pde2=self.criterion(-(grad_K2_grad_u2),f2)

        loss_boundary = (self.criterion(ubx_1, (self.bx_1[:, 0]).view_as(ubx_1)) + 
                         self.criterion(ub_1y, (self.b_1y[:,1]).view_as(ub_1y)) + 
                         self.criterion(ubx1, (self.bx1[:, 0]).view_as(ubx1) + torch.full_like(ubx1, 2)) + 
                         self.criterion(ub1y, (self.b1y[:, 1]).view_as(ub1y) + torch.full_like(ubx1, 2)))
        loss_pde1= self.criterion(-(grad_K1_grad_u1),f1)
        loss_interface = loss_jump + loss_deriv                     
        w1=0.00001
        w2=0.00001
        w3=0.99999
        w4=0.99999

        loss=w1*loss_pde1+w3*loss_interface+w2*loss_pde2+w4*loss_boundary
        loss1=w1*loss_pde1+w3*loss_interface
        loss2=w3*loss_interface+w2*loss_pde2+w4*loss_boundary
        #loss.backward()
        #print('PDE: ',loss_pde.item(),'Bound: ',loss_boundary.item(),'jmp: ',loss_jump.item(),'deriv: ',loss_deriv.item())
        return loss, loss1,loss2