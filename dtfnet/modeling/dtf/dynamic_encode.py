import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softplus
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from abc import ABC, abstractmethod
from torch_geometric.data import Data
from torch.nn import Linear, ReLU, Dropout, Parameter
from torch_geometric.nn import Sequential, GATv2Conv, JumpingKnowledge, GCNConv, GATConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor, set_diag
from dtfnet.modeling.dtf.utils import glorot, zeros
from torch_geometric.nn.dense.linear import Linear
from torch import Tensor

class SSPlus(Softplus):
    def __init__(self, beta=1, threshold=20):
        super().__init__(beta, threshold)

    def forward(self, input):
        return F.softplus(input, self.beta, self.threshold) - np.log(2.)


def gaussian(r_i, r_j, gamma: float, u_max: float, step: float):
    if u_max < 0.1:
        raise ValueError('u_max should not be smaller than 0.1')

    d = torch.linalg.vector_norm(r_i - r_j, ord=2, dim=1, keepdim=True)
    u_k = torch.arange(0, u_max, step, device=r_i.device).unsqueeze(0)
#     print(d.shape)
    out = torch.exp(-gamma * torch.square(d-u_k))
    return out

# def cos_gaussian(x_i, x_j, gamma: float, u_max: float, step: float):
#     if u_max < 0.1:
#         raise ValueError('u_max should not be smaller than 0.1')
#     import torch

#     v = F.cosine_similarity(x_i, x_j, dim=1, eps=1e-8).unsqueeze(1)
#     u_k = torch.arange(0, u_max, step, device=x_i.device).unsqueeze(0)
# #     print(v.shape)
#     out = torch.exp(-gamma * torch.square(v-u_k))
#     return out


class CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step):
        super().__init__()
        self.gamma = gamma
        self.u_max = u_max
        self.step = step

        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_g = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )

    def forward(self, x, edge_index, z, position):
        v = self.propagate(edge_index, x=x, position=position)  # 64, 256
 
        return v

    def message(self, x_i, x_j,  position_i, position_j, index):
        # g
        g = gaussian(position_i, position_j,
                     gamma=self.gamma, u_max=self.u_max, step=self.step)
        g = self.mlp_g(g)

        # out
        out = x_j * g  # 1104, 256

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out
    

# class COS_CFConv(MessagePassing):
#     def __init__(self, n_filters, gamma, u_max, step, **kwargs):
#         super().__init__(node_dim=0, **kwargs)
#         self.gamma = gamma
#         self.u_max = u_max
#         self.step = step
#         self.filters = n_filters

#         n = int(u_max / step)  # number of gaussian radial basis function
#         self.mlp_g = nn.Sequential(
#             nn.Linear(n, n_filters),
#             SSPlus(),
#             nn.Linear(n_filters, n_filters),
#             SSPlus()
#         )
        
#         self.lin_src = Linear(n_filters, n_filters, bias=False, weight_initializer='glorot')
#         self.lin_dst = self.lin_src
        
#         # 计算注意力需要的权重参数 W
#         self.att_src = Parameter(torch.Tensor(1, 1, n_filters))
#         self.att_dst = Parameter(torch.Tensor(1, 1, n_filters))
#         self.dropout = 0.0
        
#         self.bias = Parameter(torch.Tensor(n_filters))
#         self._alpha = None

#         self.reset_parameters()

#     def reset_parameters(self):
#         self.lin_src.reset_parameters()
#         self.lin_dst.reset_parameters()
#         glorot(self.att_src)
#         glorot(self.att_dst)
#         zeros(self.bias)


#     def forward(self, x, edge_index, z, position):
        
#         H, C = 1, self.filters
#         size = None
#         assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
#         x_src = x_dst = self.lin_src(x).view(-1, H, C)

#         x = (x_src, x_dst)

#         # 接下来计算节点级别的注意力系数，源节点和目标节点都需要计算
#         # 计算公式为 a^T @ x_i
#         alpha_src = (x_src * self.att_src).sum(dim=-1)
#         alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
#         alpha = (alpha_src, alpha_dst)

#         if isinstance(edge_index, Tensor):
#             # We only want to add self-loops for nodes that appear both as
#             # source and target nodes:
#             num_nodes = x_src.size(0)
#             if x_dst is not None:
#                 num_nodes = min(num_nodes, x_dst.size(0))
#             num_nodes = min(size) if size is not None else num_nodes
#             edge_index, _ = remove_self_loops(edge_index)
#             edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
#         elif isinstance(edge_index, SparseTensor):
#             edge_index = set_diag(edge_index)

# #         alpha = self.edge_updater(edge_index, alpha=alpha, edge_attr=edge_attr)
#         # propagate_type: (x: PairTensor, edge_attr: OptTensor)
#         out = self.propagate(edge_index, x=x, alpha=alpha, position=position,size=None)
        
#         alpha = self._alpha
#         self._alpha = None
#         out = out.mean(dim=1)

#         if self.bias is not None:
#             out += self.bias

#         return out


#     def message(self, x_i, x_j, alpha_i, alpha_j, position_i, position_j, index, ptr, size_i):
#         # g
#         g = gaussian(position_i, position_j, gamma=self.gamma, u_max=self.u_max, step=self.step)
#         g = self.mlp_g(g)

#         alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
#         v = F.leaky_relu(alpha, 0.2)
#         v = softmax(v, index, ptr, size_i)
#         self._alpha = v  # Save for later use.
#         v = F.dropout(v, p=self.dropout, training=self.training)
#         w = g + v
# #         print(g.shape,v.shape)
#         # out
#         out = x_j * w
#         return out
    
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.filters}, '
#                 f'{self.filters}, heads={1})')
    
#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
#         return out
    
# class COS_CFConv(MessagePassing):
#     def __init__(self, n_filters, gamma, u_max, step, **kwargs):
#         super().__init__(node_dim=0, **kwargs)
#         self.gamma = gamma
#         self.u_max = u_max
#         self.step = step
#         self.filters = n_filters

#         n = int(u_max / step)  # number of gaussian radial basis function
#         self.mlp_g = nn.Sequential(
#             nn.Linear(n, n_filters),
#             SSPlus(),
#             nn.Linear(n_filters, n_filters),
#             SSPlus()
#         )
        
#         self.att = Parameter(torch.Tensor(1, 1, n_filters))
#         self._alpha = None
#         self.dropout = 0.0
#         self._lambda = 0.0001
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.att)


#     def forward(self, x, edge_index, z, position):

#         out = self.propagate(edge_index, x=x, position=position, size=None)   # 64, 1104, 256 
#         self._alpha = None

#         return out


#     def message(self, x_i, x_j, position_i, position_j, index, ptr, size_i):

#         d = torch.linalg.vector_norm(position_i - position_j, ord=2, dim=1, keepdim=True)
#         x = (x_i.unsqueeze(1) + x_j.unsqueeze(1))
#         x = F.leaky_relu(x, 0.2)
#         alpha = (x * self.att).sum(dim=-1)
#         alpha = softmax(alpha, index, ptr, size_i)
#         self._alpha = alpha
#         v = F.dropout(alpha, p=self.dropout, training=self.training)  # 1104, 256
#         D = (1.0 - self._lambda * v) * d
        
#         u_k = torch.arange(0, self.u_max, self.step, device=position_i.device).unsqueeze(0)
#         g = torch.exp(-self.gamma * torch.square(D - u_k))
#         g = self.mlp_g(g)
#         out = x_j * g
#         return out
    
#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
#         return out

# class COS_CFConv(MessagePassing):
#     def __init__(self, n_filters, gamma, u_max, step, **kwargs):
#         super().__init__(node_dim=0, **kwargs)
#         self.gamma = gamma
#         self.u_max = u_max
#         self.step = step
#         self.filters = n_filters

#         n = int(u_max / step)  # number of gaussian radial basis function
#         self.mlp_g = nn.Sequential(
#             nn.Linear(n, n_filters),
#             SSPlus(),
#             nn.Linear(n_filters, n_filters),
#             SSPlus()
#         )
        
#         self.att = Parameter(torch.Tensor(1, 1, n_filters))
#         self._alpha = None
#         self.dropout = 0.0
#         self._lambda = 0.1
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.att)


#     def forward(self, x, edge_index, z, position):

#         out = self.propagate(edge_index, x=x, z=z, position=position, size=None)   # 64, 1104, 256 
#         self._alpha = None

#         return out


#     def message(self, x_i, x_j, z_i, z_j, position_i, position_j, index, ptr, size_i):

#         d = torch.linalg.vector_norm(position_i - position_j, ord=2, dim=1, keepdim=True)
#         v = F.cosine_similarity(z_j, z_i).unsqueeze(1)
# #         print(v.shape)
#         D = (1.0 - self._lambda * v) * d
        
#         u_k = torch.arange(0, self.u_max, self.step, device=position_i.device).unsqueeze(0)
#         g = torch.exp(-self.gamma * torch.square(D - u_k))
#         g = self.mlp_g(g)
#         out = x_j * g
#         return out
    
#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
#         return out

class COS_CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step, **kwargs):
        super().__init__(node_dim=0, **kwargs)
        self.gamma = gamma
        self.u_max = u_max
        self.step = step
        self.filters = n_filters

        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_g = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )
        
        self.mlp_v = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )
        
        self.att = Parameter(torch.Tensor(1, 1, n_filters))
        self._alpha = None
        self.dropout = 0.0
        self._lambda = 0.1
        self.z = Parameter(torch.Tensor(1))
        
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)


    def forward(self, x, edge_index, z, position):

        out = self.propagate(edge_index, x=x, z=z, position=position, size=None)   # 64, 1104, 256 
        self._alpha = None

        return out


    def message(self, x_i, x_j, z_i, z_j, position_i, position_j, index, ptr, size_i):

        d = torch.linalg.vector_norm(position_i - position_j, ord=2, dim=1, keepdim=True)
        v = F.cosine_similarity(z_j, z_i).unsqueeze(1)
        u_v = torch.arange(0, 1, 0.02, device=position_i.device).unsqueeze(0)
        g_v = torch.exp(-self.gamma * torch.square(v - u_v))
        g_v = self.mlp_v(g_v)

        
        u_k = torch.arange(0, self.u_max, self.step, device=position_i.device).unsqueeze(0)
        g = torch.exp(-self.gamma * torch.square(D - u_k))
        g = self.mlp_g(g)
        
        out = x_j * (g * g_v)
        
        return out
    
    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out

# class COS_CFConv(MessagePassing):
#     def __init__(self, n_filters, gamma, u_max, step, **kwargs):
#         super().__init__(node_dim=0, **kwargs)
#         self.gamma = gamma
#         self.u_max = u_max
#         self.step = step
#         self.filters = n_filters

#         n = int(u_max / step)  # number of gaussian radial basis function
#         self.mlp_g = nn.Sequential(
#             nn.Linear(n, n_filters),
#             SSPlus(),
#             nn.Linear(n_filters, n_filters),
#             SSPlus()
#         )
        
#         self.mlp_z = nn.Sequential(
#             nn.Linear(n_filters, n_filters),
#             SSPlus(),
#             nn.Linear(n_filters, n_filters),
#             SSPlus()
#         )
        
#         self.att = Parameter(torch.Tensor(1, 1, n_filters))
#         self._alpha = None
#         self.dropout = 0.0
#         self._lambda = 0.1
        
#         self.reset_parameters()

#     def reset_parameters(self):
#         glorot(self.att)


#     def forward(self, x, edge_index, z, position):

#         out = self.propagate(edge_index, x=x, z=z, position=position, size=None)   # 64, 1104, 256 
#         self._alpha = None

#         return out


#     def message(self, x_i, x_j, z_i, z_j, position_i, position_j, index, ptr, size_i):

#         g = gaussian(position_i, position_j, gamma=self.gamma, u_max=self.u_max, step=self.step)
#         g = self.mlp_g(g)
        
#         len_i = torch.norm(z_i, dim=1).unsqueeze(1)
#         len_j = torch.norm(z_j, dim=1).unsqueeze(1)
#         z_h = z_j * z_i
#         z = self.mlp_z(z_h)

#         w = g * z

#         # out
#         out = x_j * w
#         return out
    
#     def aggregate(self, inputs, index, ptr=None, dim_size=None):
#         out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
#         return out
    
class D_CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step):
        super().__init__()
        self.gamma = gamma
        self.u_max = u_max
        self.step = step

        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_g = nn.Sequential(
            nn.Linear(n, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )

    def forward(self, x, edge_index, z, position):
        v = self.propagate(edge_index, x=x, position=position)
        return v

    def message(self, x_i, x_j,  position_i, position_j, index):
        # g
        d = torch.linalg.vector_norm(position_i - position_j, ord=2, dim=1, keepdim=True)
        d = F.normalize(d, dim=0)
        w = 1 / (d + 1)
        out = x_j * w

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out
    
class MLP_CFConv(MessagePassing):
    def __init__(self, n_filters, gamma, u_max, step):
        super().__init__()
        self.gamma = gamma
        self.u_max = u_max
        self.step = step

        n = int(u_max / step)  # number of gaussian radial basis function
        self.mlp_d = nn.Sequential(
            nn.Linear(1, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters),
            SSPlus()
        )

    def forward(self, x, edge_index, z, position):
        v = self.propagate(edge_index, x=x, position=position)
        return v

    def message(self, x_i, x_j,  position_i, position_j, index):
        # g
        d = torch.linalg.vector_norm(position_i - position_j, ord=2, dim=1, keepdim=True)   
        w = self.mlp_d(d)
        out = x_j * w

        return out

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        out = scatter(inputs, index, dim=self.node_dim, dim_size=dim_size, reduce='sum')
        return out

class Interaction(nn.Module):
    def __init__(self,
                 conv_module,
                 n_filters: int,
                 u_max: float,
                 step: float,
                 gamma: float = 25.0,
                 ):
        super().__init__()

        self.lin_1 = nn.Linear(n_filters, n_filters)
        self.mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )

        # initialize a cfconv block
        self.cfconv = conv_module(n_filters=n_filters, gamma=gamma, u_max=u_max, step=step)

    def forward(self, x, edge_index, z, position):
        # x
        m = self.lin_1(x)
        v = self.cfconv(m, edge_index, z, position)
        v = self.mlp(v)
        x = x + v
        return x


class DynamicNet(nn.Module, ABC):
    def __init__(self, mode, n_filters, n_interactions, u_max, step, output_dim):
        super().__init__()

        self.n_interactions = n_interactions
        self.convs = nn.ModuleList()

        # Interaction module
        if mode == 'gauss':
            for _ in range(self.n_interactions):
                self.convs.append(
                Interaction(
                    conv_module=CFConv,
                    n_filters=n_filters,
                    u_max=u_max,
                    step=step
                    )
                )
        elif mode == 'cos_gauss':
            for _ in range(self.n_interactions):
                self.convs.append(
                Interaction(
                    conv_module=COS_CFConv,
                    n_filters=n_filters,
                    u_max=u_max,
                    step=step
                    )
                )
        elif mode == 'd':
            for _ in range(self.n_interactions):
                self.convs.append(
                Interaction(
                    conv_module=D_CFConv,
                    n_filters=n_filters,
                    u_max=u_max,
                    step=step
                    )
                )
        elif mode == 'mlp':
            for _ in range(self.n_interactions):
                self.convs.append(
                Interaction(
                    conv_module=MLP_CFConv,
                    n_filters=n_filters,
                    u_max=u_max,
                    step=step
                    )
                )
        else:
            raise NotImplementedError
            
        self.relu = ReLU(inplace=True)
        
        self.post_mlp = nn.Sequential(
            nn.Linear(n_filters, n_filters),
            SSPlus(),
            nn.Linear(n_filters, n_filters)
        )
        
#         self.post_mlp = Linear(n_filters, n_filters)

    def forward(self, data):
        x, edge_index, position, batch = data.x.float(), data.edge_index.long(), data.pos.float(), data.batch
        z = x
        # interaction block
        for i in range(self.n_interactions):
            x = self.convs[i](x=x, edge_index=edge_index, z=z, position=position)
#             x = self.relu(x)

        # post mlp
        x = self.post_mlp(x)

        return x
    


# class GATNet(nn.Module, ABC):
#     def __init__(self, h_size=256):
#         super().__init__()

#         self.model = Sequential('x, edge_index', [
#                                 (GATv2Conv(h_size, h_size), 'x, edge_index -> x'),
#                                 ReLU(inplace=True),
#                                 (GATv2Conv(h_size, h_size), 'x, edge_index -> x'),
#                                 ReLU(inplace=True),
#                                 Linear(h_size, h_size),
#                                 ])

#     def forward(self, data):
#         x, edge_index = data.x.float(), data.edge_index.long()

#         x= self.model(x, edge_index)

#         return x
    
class GATNet(nn.Module, ABC):
    def __init__(self, h_size=256):
        super().__init__()

        self.model = Sequential('x, edge_index', [
                                (GATConv(h_size, h_size), 'x, edge_index -> x'),
                                ReLU(inplace=True),
                                (GATConv(h_size, h_size), 'x, edge_index -> x'),
                                ReLU(inplace=True),
                                Linear(h_size, h_size),
                                ])

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.long()

        x= self.model(x, edge_index)

        return x
    
    
class GCNNet(nn.Module, ABC):
    def __init__(self, h_size=256):
        super().__init__()

        self.model = Sequential('x, edge_index', [
                                (GCNConv(h_size, h_size), 'x, edge_index -> x'),
                                ReLU(inplace=True),
                                (GCNConv(h_size, h_size), 'x, edge_index -> x'),
                                ReLU(inplace=True),
                                Linear(h_size, h_size),
                                ])

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index.long()

        x= self.model(x, edge_index)

        return x
    

def bulid_dynamicnet(cfg):
    n_filters = cfg.MODEL.DTF.JOINT_SPACE_SIZE  
    n_interactions = cfg.SOLVER.GNN_LAYERS
    u_max = cfg.SOLVER.GNN_U
    step = cfg.SOLVER.GNN_STEP
    
    if cfg.SOLVER.GNN_MODE in ['gauss', 'd', 'mlp', 'cos_gauss']:
        model = DynamicNet(mode=cfg.SOLVER.GNN_MODE, n_filters=n_filters, n_interactions=n_interactions, u_max=u_max, step=step, output_dim=1)
    elif cfg.SOLVER.GNN_MODE == 'gcn':
        model = GCNNet(h_size=n_filters)
    elif cfg.SOLVER.GNN_MODE == 'gat':
        model = GATNet(h_size=n_filters)
    else:
        raise NotImplementedError
    
    return model


if __name__ == '__main__':
    num_nodes = 64
    row, col = np.indices((num_nodes, num_nodes))
    edge_index = np.concatenate((row.reshape(-1, 1), col.reshape(-1, 1)), axis=1)
    # 按要求设置起始节点小于结束节点才有边存在
    edge_index = edge_index[(edge_index[:, 0] < edge_index[:, 1])]
    edge_index = torch.from_numpy(edge_index.T)
    # print(torch.arange(0, 64))

    data = Data(x=torch.rand((64, 256)), edge_index=edge_index, pos=torch.arange(0, 64).resize(64, 1))  # torch.randint(0, 10, (2, 50))  torch.randn(10, 1)
    model = SchNetAvg()
    # print(model)
    out = model(data)
    print(out.shape)