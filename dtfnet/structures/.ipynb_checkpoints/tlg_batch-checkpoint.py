from dataclasses import dataclass
import torch

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

# temporal localization grounding 
@dataclass
class TLGBatch(object):
    # frames: list # [ImageList]
    feats: torch.tensor 
    queries: list
    wordlens: list
    all_iou2d: list
    moments: list
    num_sentence: list
    idxs: torch.tensor
    vid: str

    def to(self, device):
        # self.frames = [f.to(device) for f in self.frames]
        self.feats = self.feats.to(device)
        self.queries = [query.to(device) for query in self.queries]
        self.wordlens = [word_len.to(device) for word_len in self.wordlens]
        self.all_iou2d = [iou2d.to(device) for iou2d in self.all_iou2d]
        self.moments = [moment.to(device) for moment in self.moments]
        self.idxs = move_to_cuda(self.idxs)
        self.vid = move_to_cuda(self.vid)

        return self
    

