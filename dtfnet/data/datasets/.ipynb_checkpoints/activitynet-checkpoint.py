import json
import logging
import torch
from .utils import  moment_to_iou2d, bert_embedding, get_vid_feat
from transformers import DistilBertTokenizer
import os
import random

class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(self, audio_dir, ann_file, feat_file, num_pre_clips, num_clips):
        super(ActivityNetDataset, self).__init__()
        self.num_pre_clips = num_pre_clips
        self.num_clips = num_clips
        self.audio_dir = audio_dir
        self.feat_file = feat_file
        with open(ann_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)

        self.annos = {}
        self.data = list(annos.keys())
        
        logger = logging.getLogger("dtf.trainer")
        if 'train' in ann_file:
            self.mode = 'train'
        if 'val' in ann_file:
            self.mode = 'val'
        if 'test' in ann_file:
            self.mode = 'test'
            
        temp = {}
        c = 0
#         vid_all = []
#         duration_all = []
#         times = []
#         audios_all = []
#         sents = []
        
        if 'train' in ann_file:
            for vid in self.data:
                duration, timestamps, audios, sentences = annos[vid]['duration'], annos[vid]['timestamps'], annos[vid]['audios'], annos[vid]['sentences']
                temp_l = len(timestamps)
            
            
                for i in range(temp_l):
                    c = c + 1
                    f_k = f"{c}"
                    timestamps_new, audios_new, sentences_new = [timestamps[i]], [audios[i]],[sentences[i]]
                    temp[f_k] ={
                        'vid': vid,
                        'duration': duration,
                        'timestamps': timestamps_new,
                        'audios': audios_new,
                        'sentences': sentences_new
                        }
            keys = list(temp.keys())
            values = list(temp.values())

            # 使用 sample 函数打乱键的顺序
            random_keys = random.sample(keys, len(keys))

            # 根据打乱后的键重构字典
            self.annos = {k: temp[k] for k in random_keys}
            
#                     vid_all.append(vid)
#                     duration_all.append(duration)
#                     times.append(timestamps_new)
#                     audios_all.append(audios_new)
#                     sents.append(sentences_new)
                    
#             index = list(range(len(vid_all)))  # 创建包含索引值的列表
#             random.shuffle(index)  # 打乱索引值的顺序

#             vid_all_new = [vid_all[i] for i in index]
#             duration_all_new = [duration_all[i] for i in index]
#             times_new = [times[i] for i in index]
#             audios_all_new = [audios_all[i] for i in index]
#             sents_new = [sents[i] for i in index]
            
            
#             for i in range(len(vid_all)):
#                 c = c + 1
#                 f_k = f"{c}"
#                 self.annos[f_k] ={
#                     'vid': vid_all_new[i],
#                     'duration': duration_all_new[i],
#                     'timestamps': times_new[i],
#                     'audios': audios_all_new[i],
#                     'sentences': sents_new[i]
#                     }                
                    
        elif 'test' in ann_file: 
            for vid in self.data:
                duration, timestamps, audios, sentences = annos[vid]['duration'], annos[vid]['timestamps'], annos[vid]['audios'], annos[vid]['sentences']
                c = c + 1
                f_k = f"{c}"
                self.annos[f_k] ={
                    'vid': vid,
                    'duration': duration,
                    'timestamps': timestamps,
                    'audios': audios,
                    'sentences': sentences
                    }
                
        self.annos_key = list(self.annos.keys())
            
#         self.moment_list = []
#         self.ious = []
#         self.audios_list = []
#         self.sent_list = []
#         for vid in self.data:
#             duration, timestamps, audios_name, sentences = annos[vid]['num_frames']/annos[vid]['fps'], annos[vid]['timestamps'], \
#                                                annos[vid]['audios'], annos[vid]['sentences']
#             moments = []
#             all_iou2d = []
#             for timestamp in timestamps:
#                 time = torch.Tensor([max(timestamp[0]/annos[vid]['fps'], 0), min(timestamp[1]/annos[vid]['fps'], duration)])
#                 iou2d = moment_to_iou2d(time, self.num_clips, duration)
#                 moments.append(time)
#                 all_iou2d.append(iou2d)
#             moments = torch.stack(moments)
#             all_iou2d = torch.stack(all_iou2d)

#             assert moments.size(0) == all_iou2d.size(0)

#             num_audios = len(audios_name)
#             if num_audios == 1:
#                 audios = torch.load(os.path.join(self.audio_dir, f'{audios_name[0].split(".")[0]}.pt')).squeeze(
#                     dim=1).float()
#             elif num_audios > 1:
#                 audios = [torch.load(os.path.join(self.audio_dir, f'{audio_name.split(".")[0]}.pt')).squeeze(dim=1).float()
#                           for audio_name in audios_name]
#                 audios = torch.squeeze(torch.stack(audios, dim=0), dim=1)
#             else:
#                 raise ValueError("num_audios should be greater than 0!")

#             assert moments.size(0) == audios.size(0)
#             start_time = moments[:, 0]
#             index = torch.argsort(start_time)

#             audios = torch.index_select(audios, dim=0, index=index)
#             moments = torch.index_select(moments, dim=0, index=index)
#             all_iou2d = torch.index_select(all_iou2d, dim=0, index=index)

#             sent = [sentences[i] for i in index]
#             self.sent_list.append(sent)
#             self.audios_list.append(audios)
#             self.moment_list.append(moments)
#             self.ious.append(all_iou2d)

        

        logger.info("-" * 60)
        logger.info(f"Preparing {len(self.annos)} {self.mode} data, please wait...")

    def __getitem__(self, idx):
        
        f_k = self.annos_key[idx]
        vid = self.annos[f_k]['vid']
        
        duration, timestamps, audios_name, sentences = self.annos[f_k]['duration'], self.annos[f_k]['timestamps'], \
                                               self.annos[f_k]['audios'], self.annos[f_k]['sentences']
        


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
            audios = torch.load(os.path.join(self.audio_dir, f'{audios_name[0].split(".")[0]}.pt')).squeeze(dim=1).float()
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
#         sent = [sentences[i] for i in index]

        return feat, audios, all_iou2d, moments, num_audios, idx, vid

    def __len__(self):
        return len(self.annos)

    def get_duration(self, idx):
        f_k = self.annos_key[idx]
        return self.annos[f_k]['duration']

    def get_sentence(self, idx):
        f_k = self.annos_key[idx]
        
        duration, timestamps, audios_name = self.annos[f_k]['duration'], self.annos[f_k]['timestamps'], \
            self.annos[f_k]['audios']
        
        sentence = self.annos[f_k]['sentences']
        moment = [torch.tensor([max(x[0], 0), min(x[1], duration)]) for x in timestamps]
        moments = torch.stack(moment)
        start_time = moments[:,0]
        index = torch.argsort(start_time)
        
        sent = [sentence[i] for i in index]

        return sent

    def get_moment(self, idx):
        f_k = self.annos_key[idx]
        vid = self.annos[f_k]['vid']
        duration, timestamps, audios_name = self.annos[f_k]['duration'], self.annos[f_k]['timestamps'], \
            self.annos[f_k]['audios']
        
#         sentence = self.annos[vid]['sentences']
        moment = [torch.tensor([max(x[0], 0), min(x[1], duration)]) for x in timestamps]
        moments = torch.stack(moment)
        start_time = moments[:,0]
        index = torch.argsort(start_time)
        moments = torch.index_select(moments, dim=0, index=index)

        return moments

    def get_vid(self, idx):
        f_k = self.annos_key[idx]
        vid = self.annos[f_k]['vid']
        return vid

    def get_iou2d(self, idx):
        f_k = self.annos_key[idx]
        vid = self.annos[f_k]['vid']
        duration, timestamps, audios_name = self.annos[f_k]['duration'], self.annos[f_k]['timestamps'], \
            self.annos[f_k]['audios']

        moment = [torch.tensor([max(x[0], 0), min(x[1], duration)]) for x in timestamps]
        moments = torch.stack(moment)
        start_time = moments[:,0]
        index = torch.argsort(start_time)
        moments = torch.index_select(moments, dim=0, index=index)
        
        all_iou2d = torch.index_select(all_iou2d, dim=0, index=index)

        return all_iou2d

    def get_num_audios(self, idx):
        f_k = self.annos_key[idx]
        vid = self.annos[f_k]['vid']
        duration, timestamps, audios_name = self.annos[f_k]['duration'], self.annos[f_k]['timestamps'], \
            self.annos[f_k]['audios']
        return len(audios_name)