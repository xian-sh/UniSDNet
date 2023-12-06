import torch
from torch import nn

from torch.utils.checkpoint import checkpoint

def apply_to_sample(f, sample):
    if hasattr(sample, '__len__') and len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {key: _apply(value) for key, value in x.items()}
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample):
    def _move_to_cuda(tensor):
        return tensor.cuda()

    return apply_to_sample(_move_to_cuda, sample)

class TextOut(nn.Module):
    def __init__(self, input_size, joint_space_size, dataset):
        super().__init__()

        self.fc_out1 = nn.Linear(input_size, joint_space_size)
        self.fc_out2 = nn.Linear(input_size, joint_space_size)
        self.dataset = dataset
        self.layernorm = nn.LayerNorm(joint_space_size)

    def forward(self, txts):

        txt_feat = []
        txt_feat_iou = []

        for txt in txts:  # each sample (several sentences) in a batch (of videos)

            query = move_to_cuda(txt)
#             query = self.layernorm(query)
            out_iou = self.fc_out1(query)
            out = self.fc_out2(query)

            txt_feat.append(out.squeeze(0))
            txt_feat_iou.append(out_iou.squeeze(0))

        return txt_feat, txt_feat_iou


def build_text_out(cfg):
    joint_space_size = cfg.MODEL.DTF.JOINT_SPACE_SIZE
    dataset_name = cfg.DATASETS.NAME
    return TextOut(joint_space_size, joint_space_size, dataset_name)


if __name__ == "__main__":

    from mmn.config import cfg

    model = build_audio_encoder(cfg)
    model = model.cuda()
    model.eval()


