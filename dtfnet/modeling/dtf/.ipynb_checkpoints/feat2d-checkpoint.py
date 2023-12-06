import torch
from torch import nn
class SparseMaxPool(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
        return map2d

class SparseMaxPool_B(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool_B, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1

        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        src=x
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            map2d[:, :, i, j] = x
            s_f = src[:, :, i]
            e_f = src[:, :, j]
            map2d[:, :, i, j] += (s_f + e_f)
                    
                    
        return map2d
    
class SparseMaxPool_C(nn.Module):
    def __init__(self, pooling_counts, N):
        super(SparseMaxPool_C, self).__init__()
        
        self.vis_conv = nn.Conv2d(256*3, 256, 1, 1)
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2
        
        poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
        for c in pooling_counts[1:]:
            poolers.extend(
                [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij
        self.poolers = poolers

    def forward(self, x):
        src=x
        B, D, N = x.shape
        map2d = x.new_zeros(B, D*3, N, N)
        map2d[:, :, range(N), range(N)] = x.repeat(1, 3, 1)  # fill a diagonal line
        for pooler, (i, j) in zip(self.poolers, self.maskij):
            x = pooler(x)
            s_f = src[:, :, i]
            e_f = src[:, :, j]
            map2d[:, :, i, j] = torch.cat((s_f, x, e_f), dim=1)  
            
        map2d_c = self.vis_conv(map2d)      
        return map2d_c


class SparseConv(nn.Module):
    def __init__(self, pooling_counts, N, hidden_size):
        super(SparseConv, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        src = x
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
            map2d[:, :, i, j] = x
            
        return map2d


class SparseConv_B(nn.Module):
    def __init__(self, pooling_counts, N, hidden_size):
        super(SparseConv_B, self).__init__()
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        src = x
        B, D, N = x.shape
        map2d = x.new_zeros(B, D, N, N)
        map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
#             map2d[:, :, i, j] = x
            s_f = src[:, :, i]
            e_f = src[:, :, j]
            map2d[:, :, i, j] += (s_f + e_f)         
                    
        return map2d
    
class SparseConv_C(nn.Module):
    def __init__(self, pooling_counts, N, hidden_size):
        super(SparseConv_C, self).__init__()
        
        self.vis_conv = nn.Conv2d(256*3, 256, 1, 1)
        mask2d = torch.zeros(N, N, dtype=torch.bool)
        mask2d[range(N), range(N)] = 1
        self.hidden_size = hidden_size
        stride, offset = 1, 0
        maskij = []
        for c in pooling_counts:
            # fill all diagonal lines
            for _ in range(c):
                # fill a diagonal line
                offset += stride
                i, j = range(0, N - offset, stride), range(offset, N, stride)
                mask2d[i, j] = 1
                maskij.append((i, j))
            stride *= 2

        self.convs = nn.ModuleList()
        self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
        for c in pooling_counts[1:]:
            self.convs.extend(
                [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
            )

        self.mask2d = mask2d.to("cuda")
        self.maskij = maskij

    def forward(self, x):
        src = x
        B, D, N = x.shape
        map2d = x.new_zeros(B, D*3, N, N)
        map2d[:, :, range(N), range(N)] = x.repeat(1, 3, 1)  # fill a diagonal line
        
        for conv, (i, j) in zip(self.convs, self.maskij):
            x = conv(x)
#             map2d[:, :, i, j] = x
            s_f = src[:, :, i]
            e_f = src[:, :, j]
            map2d[:, :, i, j] = torch.cat((s_f, x, e_f), dim=1)         
                    
        map2d_c = self.vis_conv(map2d)      
        return map2d_c
    


    
# class SparseMaxPool(nn.Module):
#     def __init__(self, input_size, pooling_counts, N):
#         super(SparseMaxPool, self).__init__()
#         mask2d = torch.zeros(N, N, dtype=torch.bool)
#         mask2d[range(N), range(N)] = 1

#         stride, offset = 1, 0
#         maskij = []
#         for c in pooling_counts:
#             # fill all diagonal lines
#             for _ in range(c):
#                 # fill a diagonal line
#                 offset += stride
#                 i, j = range(0, N - offset, stride), range(offset, N, stride)
#                 mask2d[i, j] = 1
#                 maskij.append((i, j))
#             stride *= 2
        
#         poolers = [nn.MaxPool1d(2, 1) for _ in range(pooling_counts[0])]
#         for c in pooling_counts[1:]:
#             poolers.extend(
#                 [nn.MaxPool1d(3, 2)] + [nn.MaxPool1d(2, 1) for _ in range(c - 1)]
#             )

#         self.mask2d = mask2d.to("cuda")
#         self.maskij = maskij
#         self.poolers = poolers
#         self.conv_layer = nn.Conv2d(input_size * 2, input_size, 1, 1)
#         self.relu = nn.ReLU(True)
        
#     def forward(self, x):
#         src = x
#         B, D, N = x.shape
#         map2d = x.new_zeros(B, D, N, N).cuda()
#         boundary_map2d = x.new_zeros(B, D, N, N).cuda()
#         map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
#         boundary_map2d[:, :, range(N), range(N)] = x
#         for pooler, (i, j) in zip(self.poolers, self.maskij):
#             x = pooler(x)
#             map2d[:, :, i, j] = x
        
#         for i in range(N):
#             for j in range(i+1, N):
#                 if self.mask2d[i, j] == 1:
#                     s_f = src[:, :, i]
#                     e_f = src[:, :, j]
#                     boundary_map2d[:, :, i, j] = (s_f + e_f)/2
            
# #         fused_map = torch.cat((boundary_map2d, map2d), dim=1)
# # #         print(fused_map.shape)
# #         out = self.relu(self.conv_layer(fused_map))
        
#         return map2d, boundary_map2d

# class SparseConv(nn.Module):
#     def __init__(self, input_size, pooling_counts, N, hidden_size):
#         super(SparseConv, self).__init__()
#         mask2d = torch.zeros(N, N, dtype=torch.bool)
#         mask2d[range(N), range(N)] = 1
#         self.hidden_size = hidden_size
#         stride, offset = 1, 0
#         maskij = []
#         for c in pooling_counts:
#             # fill all diagonal lines
#             for _ in range(c):
#                 # fill a diagonal line
#                 offset += stride
#                 i, j = range(0, N - offset, stride), range(offset, N, stride)
#                 mask2d[i, j] = 1
#                 maskij.append((i, j))
#             stride *= 2

#         self.convs = nn.ModuleList()
#         self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
#         for c in pooling_counts[1:]:
#             self.convs.extend(
#                 [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
#             )

#         self.mask2d = mask2d.to("cuda")
#         self.maskij = maskij
#         self.conv_layer = nn.Conv2d(input_size * 2, input_size, 1, 1)
#         self.relu = nn.ReLU(True)

#     def forward(self, x):
#         src = x
#         B, D, N = x.shape
#         map2d = x.new_zeros(B, D, N, N).cuda()
#         boundary_map2d = x.new_zeros(B, D, N, N).cuda()
#         map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
#         boundary_map2d[:, :, range(N), range(N)] = x
#         for conv, (i, j) in zip(self.convs, self.maskij):
#             x = conv(x)
#             map2d[:, :, i, j] = x
            
#         for i in range(N):
#             for j in range(i+1, N):
#                 if self.mask2d[i, j] == 1:
#                     s_f = src[:, :, i]
#                     e_f = src[:, :, j]
#                     boundary_map2d[:, :, i, j] = (s_f + e_f)/2
                    
# #         for (i, j) in self.maskij:
# #             boundary_map2d[:, :, i, j] = (x[:, :, i] + x[:, :, j]) / 2
            
# #         fused_map = torch.cat((boundary_map2d, map2d), dim=1)
# # #         print(fused_map.shape)
# #         out = self.relu(self.conv_layer(fused_map))
        
#         return map2d, boundary_map2d
    
# class SparseConv(nn.Module):
#     def __init__(self, pooling_counts, N, hidden_size):
#         super(SparseConv, self).__init__()
#         mask2d = torch.zeros(N, N, dtype=torch.bool)
#         mask2d[range(N), range(N)] = 1
#         self.hidden_size = hidden_size
#         stride, offset = 1, 0
#         maskij = []
#         for c in pooling_counts:
#             # fill all diagonal lines
#             for _ in range(c):
#                 # fill a diagonal line
#                 offset += stride
#                 i, j = range(0, N - offset, stride), range(offset, N, stride)
#                 mask2d[i, j] = 1
#                 maskij.append((i, j))
#             stride *= 2

#         self.convs = nn.ModuleList()
#         self.convs.extend([nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(pooling_counts[0])])
#         for c in pooling_counts[1:]:
#             self.convs.extend(
#                 [nn.Conv1d(hidden_size, hidden_size, 3, 2)] + [nn.Conv1d(hidden_size, hidden_size, 2, 1) for _ in range(c - 1)]
#             )

#         self.mask2d = mask2d.to("cuda")
#         self.maskij = maskij

#     def forward(self, x):
#         B, D, N = x.shape
#         map2d = x.new_zeros(B, D, N, N)
#         map2d[:, :, range(N), range(N)] = x  # fill a diagonal line
#         for conv, (i, j) in zip(self.convs, self.maskij):
#             x = conv(x)
#             map2d[:, :, i, j] = x
#         return map2d


# def build_feat2d(cfg):
#     input_size = cfg.MODEL.DTF.JOINT_SPACE_SIZE
#     pooling_counts = cfg.MODEL.DTF.FEAT2D.POOLING_COUNTS  # [15,8,8] anet, [15] charades
#     num_clips = cfg.MODEL.DTF.NUM_CLIPS  # 64 anet, 16 charades
#     hidden_size = cfg.MODEL.DTF.FEATPOOL.HIDDEN_SIZE  # 512
#     if cfg.MODEL.DTF.FEAT2D.NAME == "conv":
#         return SparseConv(input_size, pooling_counts, num_clips, hidden_size)
#     elif cfg.MODEL.DTF.FEAT2D.NAME == "pool":
#         return SparseMaxPool(input_size, pooling_counts, num_clips)
#     else:
#         raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.DTF.FEAT2D.NAME)

def build_feat2d(cfg):
    pooling_counts = cfg.MODEL.DTF.FEAT2D.POOLING_COUNTS  # [15,8,8] anet, [15] charades
    num_clips = cfg.MODEL.DTF.NUM_CLIPS  # 64 anet, 16 charades
    hidden_size = cfg.MODEL.DTF.FEATPOOL.HIDDEN_SIZE  # 512
    if cfg.MODEL.DTF.FEAT2D.NAME == "conv":
        return SparseConv(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "pool":
        return SparseMaxPool(pooling_counts, num_clips)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "concat":
        return SparseConcat(pooling_counts, num_clips)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "conv_b":
        return SparseConv_B(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "pool_b":
        return SparseMaxPool_B(pooling_counts, num_clips)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "conv_c":
        return SparseConv_C(pooling_counts, num_clips, hidden_size)
    elif cfg.MODEL.DTF.FEAT2D.NAME == "pool_c":
        return SparseMaxPool_C(pooling_counts, num_clips)
    else:
        raise NotImplementedError("No such feature 2d method as %s" % cfg.MODEL.DTF.FEAT2D.NAME)
