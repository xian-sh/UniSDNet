import torch
from torch import nn
# from transformers import ASTModel


class DistilBert(nn.Module):
    def __init__(self, joint_space_size, dataset):
        super().__init__()

        # self.bert = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.fc_out1 = nn.Linear(768, joint_space_size)
        self.fc_out2 = nn.Linear(768, joint_space_size)
        self.dataset = dataset
        # self.layernorm = nn.LayerNorm(768)
        # self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, queries):
        '''
        Average pooling over bert outputs among words to be sentence feature
        :param queries:
        :param wordlens:
        :param vid_avg_feat: B x C
        :return: list of [num_sent, C], len=Batch_size
        '''
        sent_feat = []
        sent_feat_iou = []
        for query in queries:  # each sample (several sentences) in a batch (of videos)
            query = query.cuda()

            # x = self.bert(input_values=query)
            # x = x.last_hidden_state
            # x = self.layernorm(x)
            # x = self.avgpool(x.transpose(1, 2))
            # x = x.transpose(1, 2).squeeze(0)
            x = query
            # print(x.shape)
            out_iou = self.fc_out1(x).squeeze(0)
            out = self.fc_out2(x).squeeze(0)
            # print(out.shape)
            if out_iou.ndim == 1 and out.ndim == 1:
                out_iou = out_iou.view(1,-1)
                out = out.view(1, -1)
            
            sent_feat.append(out)
            sent_feat_iou.append(out_iou)
        return sent_feat, sent_feat_iou


def build_text_encoder(cfg):
    joint_space_size = cfg.MODEL.DTF.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    return DistilBert(joint_space_size, dataset_name)
