import torch
from torch import nn
from torch.functional import F
from dtfnet.modeling.dtf.featpool import build_featpool  # downsample 1d temporal features to desired length
from dtfnet.modeling.dtf.feat2d import build_feat2d  # use MaxPool1d/Conv1d to generate 2d proposal-level feature map from 1d temporal features
from dtfnet.modeling.dtf.loss import build_contrastive_loss
from dtfnet.modeling.dtf.loss import build_bce_loss
from dtfnet.modeling.dtf.text_encoder import build_text_encoder
from dtfnet.modeling.dtf.proposal_conv import build_proposal_conv

from dtfnet.config import cfg as config
from dtfnet.modeling.dtf.position_encoding import build_position_encoding
from torch_geometric.data import Data
from dtfnet.modeling.dtf.dynamic_encode import bulid_dynamicnet
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from dtfnet.utils.comm import move_to_cuda
from dtfnet.modeling.dtf.text_out import build_text_out

class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz).cuda()
        layers = [
            nn.Dropout(dropout).cuda(),
            nn.Linear(in_hsz, out_hsz).cuda(),
        ]
        self.net = nn.Sequential(*layers).cuda()

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x.float())
        x = self.net(x.float())
        if self.relu:
            x = F.relu(x.float(), inplace=True)
        return x  # (N, L, D)

    
class StaticTemporalNet(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos):

        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)

        src2 = self.norm1(src)
        src2 = src2 + pos
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        src = self.norm3(src)

        src = src.transpose(0, 1)  # (batch_size, L, d)

        return src

def build_staticnet(x):
    return StaticTemporalNet(d_model=x, dropout=0.1, dim_feedforward=1024)


class EasyMLP(nn.Module):
    def __init__(self, d_model=512, dim_feedforward=1024, dropout=0.1):
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, pos):

        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)


        src2 = src + pos
        src = src + self.dropout(src2)
        src2 = self.norm(src)

        src = src.transpose(0, 1)  # (batch_size, L, d)

        return src

def build_easymlp(x):
    return EasyMLP(d_model=x, dropout=0.1, dim_feedforward=1024)

def draw_map(map2d, aid_feats, save_dir, batches, filename, cmap='OrRd'):

    for i in range(len(map2d)):
        save_dir1 = os.path.join(save_dir, f'{batches.vid[i]}')
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if aid_feats == None:
            sim_matrices = map2d[i]
        else:
            v_map = map2d[i].view(256, 64*64)  # 256, 64, 64
            audio = aid_feats[i].squeeze(0)
            sim_matrices = torch.matmul(audio, v_map).view(audio.size(0), 64, 64)

        for j in range(len(sim_matrices)):
            # 绘制 heatmap 并保存图像
            fig, ax = plt.subplots(figsize=(17, 16))
            s = sim_matrices[j].squeeze().cpu().detach().numpy()
            max_index = np.unravel_index(np.argmax(s), s.shape)
            im = ax.imshow(s, cmap=cmap, interpolation='nearest', aspect='auto', norm=colors.Normalize(vmin=0, vmax=np.max(s, axis=None)))
            ax.set_yticks(range(64))
            ax.set_xticks(range(64))
            ax.set_ylabel('start index', fontsize=12)
            ax.set_xlabel('end index', fontsize=12)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.scatter(max_index[1], max_index[0], marker='*', color='lightgreen', s=1000)
            x = filename.format(j)
            filepath = os.path.join(save_dir1, x)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0, dpi=300)
            plt.clf()
            

class DTF(nn.Module):
    def __init__(self, cfg):
        super(DTF, self).__init__()
        

        # config
        self.cfg = cfg
        self.use_static = cfg.SOLVER.USE_STATIC
        self.use_gnn = cfg.SOLVER.USE_GNN
        self.gnn_sparse = cfg.SOLVER.GNN_SPARSE
        self.gnn_mode = cfg.SOLVER.GNN_MODE        # 'gauss' or 'd' or 'gat'
        self.pos_embed = cfg.SOLVER.POS_EMBED      # choices=['trainable', 'sine', 'learned']
        
        # other 
        self.joint_space = cfg.MODEL.DTF.JOINT_SPACE_SIZE
        
        # video
        self.featpool = build_featpool(cfg)
        self.feat2d = build_feat2d(cfg)
        
        # audio 
        self.encoder_name = cfg.MODEL.DTF.TEXT_ENCODER.NAME
        self.text_encoder = build_text_encoder(cfg)
        self.text_out = build_text_out(cfg)

        # use static
        if self.use_static:
            self.static_net = build_staticnet(self.joint_space).cuda()
            
        # use gnn
        if self.use_gnn:
            self.gnn = bulid_dynamicnet(cfg)
            
            N = cfg.MODEL.DTF.NUM_CLIPS
            if self.gnn_sparse == False:
                mask2d = torch.zeros([N, N])
                mask2d = mask2d.triu(diagonal=0)
            elif self.gnn_sparse:
                mask2d = self.feat2d.mask2d
            edge_indices = torch.nonzero(mask2d)  # (N, 2)
            row_indices = edge_indices[:, 0]
            col_indices = edge_indices[:, 1]
            self.edge_index = torch.stack((row_indices, col_indices), dim=0).cuda()  # (2, N)
            self.node_pos = torch.arange(0, N).view(N, 1).cuda()
            
            self.mlp = build_easymlp(self.joint_space).cuda()
            
            
        # 只要使用其一就需要预处理，映射
        if self.use_static or self.use_gnn:
            
            self.vid_position_embedding = cfg.SOLVER.POS_EMBED  # choices=['trainable', 'sine', 'learned']
            self.audio_position_embedding = cfg.SOLVER.POS_EMBED  # choices=['trainable', 'sine', 'learned']
            self.vid_pos_embed, self.audio_pos_embed = build_position_encoding(self.vid_position_embedding, self.audio_position_embedding, self.joint_space)
            
#             self.audio_encoder = build_audio_encoder(self.joint_space, cfg)
            self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d, self.joint_space)
            self.hidden_dim = self.joint_space
            n_input_proj = 2
            relu_args = [True] * 3
            relu_args[n_input_proj - 1] = False

#             self.input_audio_proj = nn.Sequential(*[LinearLayer(self.audio_input, self.hidden_dim, layer_norm=True,
#                                                             dropout=0.5, relu=relu_args[0]),
#                                               LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
#                                                           dropout=0.5, relu=relu_args[1]),
#                                               LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
#                                                           dropout=0.5, relu=relu_args[2])][:n_input_proj])

            self.input_vid_proj = nn.Sequential(*[LinearLayer(cfg.MODEL.DTF.FEATPOOL.HIDDEN_SIZE, self.hidden_dim, layer_norm=True,
                                                          dropout=0.5, relu=relu_args[0]),
                                              LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
                                                          dropout=0.5, relu=relu_args[1]),
                                              LinearLayer(self.hidden_dim, self.hidden_dim, layer_norm=True,
                                                          dropout=0.5, relu=relu_args[2])][:n_input_proj])

        elif not self.use_static and not self.use_gnn:
#             self.audio_encoder = build_audio_encoder(self.audio_input, cfg)
            self.proposal_conv = build_proposal_conv(cfg, self.feat2d.mask2d, 512)
  
        
#         self.fuse = Fuse(cfg)
        self.iou_score_loss = build_bce_loss(cfg, self.feat2d.mask2d)
        self.contrastive_loss = build_contrastive_loss(cfg, self.feat2d.mask2d)
        self.only_iou_loss_epoch = cfg.SOLVER.ONLY_IOU

    def forward(self, batches, cur_epoch=1):
        """
        Arguments:
            batches.all_iou2d: list(B) num_sent x T x T
            feat2ds: B x C x T x T
            sent_feats: list(B) num_sent x C
        """
#         save_dir = self.cfg.OUTPUT_DIR + '/visual_audio/'
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
            
        # backbone
        ious2d = batches.all_iou2d
        assert len(ious2d) == batches.feats.size(0)
        for idx, (iou, sent) in enumerate(zip(ious2d, batches.queries)):
            assert iou.size(0) == sent.size(0)
            assert iou.size(0) == batches.num_sentence[idx]
            
        feats = self.featpool(batches.feats).cuda()  # B x C x T  4,512,64
        mask_feats = torch.ones(feats.size(0), 1, feats.size(2)).cuda()
        
        sent_feat, sent_feat_iou = self.text_encoder(batches.queries)
        mask_txt = [torch.ones(1, len(sent)) for sent in sent_feat]
        
        
        feats = feats.transpose(1, 2)  # B x T x C  48,16,512
#         print(feats.shape)
        sent_feat, sent_feat_iou = move_to_cuda(sent_feat), move_to_cuda(sent_feat_iou)
        mask_txt = move_to_cuda(mask_txt)

        vid_feats = []
        txt_feats = []
        if self.use_static and self.use_gnn:
            for i, (vid, txt) in enumerate(zip(feats, sent_feat), start=0):

                src_vid = self.input_vid_proj(vid.cuda()).unsqueeze(0)
                src_txt = txt.unsqueeze(0)
                
                pos_vid = self.vid_pos_embed(src_vid, mask_feats[i])  # 1, 64, 256
                pos_txt = self.audio_pos_embed(src_txt, mask_txt[i])  # 1,  num, 256

                src = torch.cat([src_vid, src_txt], dim=1)  # (batch_size, L_vid+L_txt, d)
                pos = torch.cat([pos_vid, pos_txt], dim=1)

                memory = self.static_net(src, pos)  # hs: (#layers, batch_size, #queries, d)
                
                txt_mem = memory[:, src_vid.shape[1]:]  # (batch_size, L_txt, d)
                vid_mem = memory[:, :src_vid.shape[1]]  # (batch_size, L_vid, d)   1 128 256
                
                vid_data = Data(x=vid_mem.squeeze(0), pos=self.node_pos, edge_index=self.edge_index)
                out = self.gnn(vid_data)
                vid_feats.append(out.squeeze(0))
                txt_feats.append(txt_mem.squeeze(0))

            vid_feats = torch.stack(vid_feats, dim=0).squeeze(1).transpose(1, 2)  # B  x d x L_vid
        
        elif self.use_static and not self.use_gnn:
            for i, (vid, txt) in enumerate(zip(feats, sent_feat), start=0):

                src_vid = self.input_vid_proj(vid.cuda()).unsqueeze(0)
                src_txt = txt.unsqueeze(0)
                
                pos_vid = self.vid_pos_embed(src_vid, mask_feats[i])  # 1, 64, 256
                pos_txt = self.audio_pos_embed(src_txt, mask_txt[i])  # 1,  num, 256

                src = torch.cat([src_vid, src_txt], dim=1)  # (batch_size, L_vid+L_txt, d)
                pos = torch.cat([pos_vid, pos_txt], dim=1)

                memory = self.static_net(src, pos)  # hs: (#layers, batch_size, #queries, d)
                txt_mem = memory[:, src_vid.shape[1]:]  # (batch_size, L_txt, d)
                vid_mem = memory[:, :src_vid.shape[1]]  # (batch_size, L_vid, d)

                txt_feats.append(txt_mem.squeeze(0))
                vid_feats.append(vid_mem.squeeze(0))

            vid_feats = torch.stack(vid_feats, dim=0).squeeze(1).transpose(1, 2)  # B  x d x L_vid
 
        elif not self.use_static and self.use_gnn:
            for i, (vid, txt) in enumerate(zip(feats, sent_feat), start=0):
                
                src_vid = self.input_vid_proj(vid.cuda()).unsqueeze(0)
                src_txt = txt.unsqueeze(0)
                
                pos_vid = self.vid_pos_embed(src_vid, mask_feats[i])  # 1, 64, 256
                pos_txt = self.audio_pos_embed(src_txt, mask_txt[i])  # 1,  num, 256

                src = torch.cat([src_vid, src_txt], dim=1)  # (batch_size, L_vid+L_txt, d)
                pos = torch.cat([pos_vid, pos_txt], dim=1)
                
                memory = self.mlp(src, pos)
                txt_mem = memory[:, src_vid.shape[1]:]  # (batch_size, L_txt, d)
                vid_mem = memory[:, :src_vid.shape[1]]  # (batch_size, L_vid, d)
 
                vid_data = Data(x=vid_mem.squeeze(0), pos=self.node_pos, edge_index=self.edge_index)
                out = self.gnn(vid_data)
                vid_feats.append(out.squeeze(0))
                txt_feats.append(txt_mem.squeeze(0))

            vid_feats = torch.stack(vid_feats, dim=0).squeeze(1).transpose(1, 2)  # B  x d x L_vid
            
        elif not self.use_static and not self.use_gnn:
            vid_feats = feats.transpose(1, 2)
            txt_feats = sent_feat
        
#         print(feats.shape)  #48, 16, 512
        map2d = self.feat2d(vid_feats)  # use MaxPool1d to generate 2d proposal-level feature map from 1d temporal features
        map2d, map2d_iou = self.proposal_conv(map2d)
#         print(txt_feats[0].shape)
        
        txt_feat, txt_feat_iou = self.text_out(txt_feats)
#         print(txt_feat_iou[0].shape)

        # inference
        contrastive_scores = []
        iou_scores = []
        _, T, _ = map2d[0].size()
        for i, sf_iou in enumerate(txt_feat_iou):  # sent_feat_iou: [num_sent x C] (len=B)
            # iou part
            vid_feat_iou = map2d_iou[i]  # C x T x T
            vid_feat_iou_norm = F.normalize(vid_feat_iou, dim=0)
            sf_iou = sf_iou.reshape(-1,self.joint_space)
            sf_iou_norm = F.normalize(sf_iou, dim=1)
            
#             content_map = vid_feat_iou_norm  # 1,c,t,t
#             boundary_map = F.normalize(boundary_map2d[i], dim=0)

#             score = self.fuse(boundary_map, content_map, self.feat2d.mask2d, sf_iou_norm)
            
            iou_score = torch.mm(sf_iou_norm, vid_feat_iou_norm.reshape(vid_feat_iou_norm.size(0), -1)).reshape(-1, T, T)  # num_sent x T x T
            iou_scores.append((iou_score*10).sigmoid() * self.feat2d.mask2d)
        
        iou_scores = move_to_cuda(iou_scores)
        ious2d = move_to_cuda(list(ious2d))
#         draw_map(iou_scores, None, save_dir, batches, filename='iou_scores_{}.png')
#         draw_map(ious2d, None, save_dir, batches, filename='ious2d_{}.png')
        
        # loss
        if self.training:
            loss_iou = self.iou_score_loss(torch.cat(iou_scores, dim=0), torch.cat(ious2d, dim=0), cur_epoch)
            loss_vid, loss_sent = self.contrastive_loss(map2d, txt_feats, ious2d, batches.moments)
            return loss_vid, loss_sent, loss_iou
        else:
            for i, sf in enumerate(txt_feats):
                # contrastive part
                vid_feat = map2d[i, ...]  # C x T x T
                vid_feat_norm = F.normalize(vid_feat, dim=0)
                sf_norm = F.normalize(sf, dim=1)  # num_sent x C
                _, T, _ = vid_feat.size()
                contrastive_score = torch.mm(sf_norm, vid_feat_norm.reshape(vid_feat.size(0), -1)).reshape(-1, T, T) * self.feat2d.mask2d  # num_sent x T x T
                contrastive_scores.append(contrastive_score)
            return map2d_iou, sent_feat_iou, contrastive_scores, iou_scores  # first two maps for visualization
