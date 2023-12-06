import json
import logging
import torch
from .utils import  moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer
import os

class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, ann_file, feat_file, num_pre_clips, num_clips):
        super(ActivityNetDataset, self).__init__()
        self.num_pre_clips = num_pre_clips
        self.num_clips = num_clips
        self.audio_dir = audio_dir
        self.feat_file = feat_file
        with open(ann_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)

        self.annos = annos
        self.data = list(annos.keys())
        
        logger = logging.getLogger("dtf.trainer")
        if 'train' in ann_file:
            self.mode = 'train'
        if 'val' in ann_file:
            self.mode = 'val'
        if 'test' in ann_file:
            self.mode = 'test'

        logger.info("-" * 60)
        logger.info(f"Preparing {len(self.data)} {self.mode} data, please wait...")
        
        self.feat_list = []
        self.moments_list = []
        self.all_iou2d_list = []
        self.audios_list = []
        self.num_audios_list = []
        self.sent_list = []
        for vid in self.data:
            duration, timestamps, audios_name, sentences = annos[vid]['duration'], annos[vid]['timestamps'], \
                                               annos[vid]['audios'], annos[vid]['sentences']
            feat = get_vid_feat(self.feat_file, vid, self.num_pre_clips, dataset_name="activitynet")
            moments = []
            all_iou2d = []
            for timestamp in timestamps:
                time = torch.Tensor([max(timestamp[0], 0), min(timestamp[1], duration)])
                iou2d = moment_to_iou2d(time, self.num_clips, duration)
                moments.append(time)
                all_iou2d.append(iou2d)
            moments = torch.stack(moments)
            all_iou2d = torch.stack(all_iou2d)

            assert moments.size(0) == all_iou2d.size(0)

            num_audios = len(audios_name)
            if num_audios == 1:
                audios = torch.load(os.path.join(self.audio_dir, f'{audios_name[0].split(".")[0]}.pt')).squeeze(
                    dim=1).float()
            elif num_audios > 1:
                audios = [torch.load(os.path.join(self.audio_dir, f'{audio_name.split(".")[0]}.pt')).squeeze(dim=1).float()
                          for audio_name in audios_name]
                audios = torch.squeeze(torch.stack(audios, dim=0), dim=1)
            else:
                raise ValueError("num_audios should be greater than 0!")

            assert moments.size(0) == audios.size(0)
            start_time = moments[:, 0]
            index = torch.argsort(start_time)

            audios = torch.index_select(audios, dim=0, index=index)
            moments = torch.index_select(moments, dim=0, index=index)
            all_iou2d = torch.index_select(all_iou2d, dim=0, index=index)

            sent = [sentences[i] for i in index]
            self.feat_list.append(feat)
            self.audios_list.append(audios)
            self.num_audios_list.append(num_audios)
            self.moments_list.append(moments)
            self.all_iou2d_list.append(all_iou2d)
            self.sent_list.append(sent)

        

    def __getitem__(self, idx):
        vid = self.data[idx]

        return self.feat_list[idx], self.audios_list[idx], self.all_iou2d_list[idx], self.moments_list[idx], self.num_audios_list[idx], idx, vid

    def __len__(self):
        return len(self.data)

    def get_duration(self, idx):
        vid = self.data[idx]
        return self.annos[vid]['duration']

    def get_sentence(self, idx):

        return self.sent_list[idx]

    def get_moment(self, idx):

        return self.moments_list[idx]

    def get_vid(self, idx):
        vid = self.data[idx]
        return vid

    def get_iou2d(self, idx):

        return self.all_iou2d_list[idx]

    def get_num_audios(self, idx):
        return self.num_audios_list[idx]