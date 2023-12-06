import torch
from torch.nn.utils.rnn import pad_sequence
from dtfnet.structures import TLGBatch


class BatchCollator(object):
    """
    Collect batch for dataloader
    """

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        # [xxx, xxx, xxx], [xxx, xxx, xxx] ......
        feats, queries, ious2d, moments, num_sentence, idxs, vid = transposed_batch

        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=queries,
            all_iou2d=ious2d,
            moments=moments,
            num_sentence=num_sentence,
            idxs=idxs,
            vid=vid,
        ), idxs
