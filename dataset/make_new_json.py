import json
import logging

import torch
# from mmn.data.datasets.utils import moment_to_iou2d
import torch
from torch import nn
# from transformers import AutoProcessor, Data2VecAudioModel
# from torch.utils.checkpoint import checkpoint
# import librosa
import os
from mmn.utils.comm import get_rank
from mmn.utils.logger import setup_logger
# from tqdm import tqdm


class ActivityNetDataset(torch.utils.data.Dataset):
    def __init__(self, ann_file):
        super(ActivityNetDataset, self).__init__()

        with open(ann_file, 'r', encoding='utf-8') as f:
            annos = json.load(f)
        self.data = annos

        if 'train' in ann_file:
            self.mode = 'train'
            self.data_num = 10009
        elif 'val' in ann_file:
            self.mode = 'val'
            self.data_num = 4917
        elif 'test' in ann_file:
            self.mode = 'test'
            self.data_num = 4885

        save_path: str = './ActivityNet'
        logger = setup_logger("mmn_data.save", save_path, get_rank(), filename=f"log_{self.mode}.txt")
        logger.info(f"Preparing {self.data_num} {self.mode} data, please wait...")

        self.annos = {}

        # duration_data = []
        timestamp_data = []
        sentence_data = []
        aid_data = []

        timestamp, sentence, aid = self.data[0][2], self.data[0][3], self.data[0][4]

        timestamp_data.append(timestamp)
        sentence_data.append(sentence)
        aid_data.append(aid)

        c = 0
        for i, anno in enumerate(self.data):
            vid = anno[0]
            timestamp, sentence, aid = anno[2], anno[3], anno[4]

            if i > 0:
                if timestamp[0] < timestamp[1]:

                    if vid == self.data[i - 1][0]:
                        timestamp_data.append(timestamp)
                        sentence_data.append(sentence)
                        aid_data.append(aid)

                    elif vid != self.data[i - 1][0]:
                        c = c + 1
                        logger.info(f"{c}    {self.data[i - 1][0]}")

                        self.annos.update({
                            f"{self.data[i - 1][0]}": {
                                "duration": self.data[i - 1][1],
                                "timestamps": timestamp_data,
                                "sentences": sentence_data,
                                "audios": aid_data
                            }
                        })

                        timestamp_data = []
                        sentence_data = []
                        aid_data = []
                        timestamp_data.append(timestamp)
                        sentence_data.append(sentence)
                        aid_data.append(aid)

        c = c + 1
        logger.info(f"{c}    {self.data[-1][0]}")

        self.annos.update({
            f"{self.data[-1][0]}": {
                "duration": self.data[-1][1],
                "timestamps": timestamp_data,
                "sentences": sentence_data,
                "audios": aid_data
            }
        })

        with open(f'{save_path}/{self.mode}_audio.json', 'w', encoding='utf-8') as f:
            json.dump(self.annos, f, indent=4, separators=(',', ': '))

        logger.info("-" * 20 + f'{self.mode} data saved' + "-" * 20)


if __name__ == '__main__':

    # feat_file = "H:\\Code\\SVG\\data\\activity-c3d"
    # num_pre_clips = 256
    # num_clips = 64

    # test data num 4885
    ann_file = "G:\\ActivityNet_code\\Data2vec_MMN\\dataset\\ActivityNet\\new_test_data.json"
    test_data = ActivityNetDataset(ann_file)

    # val data num 4917
    ann_file = "G:\\ActivityNet_code\\Data2vec_MMN\\dataset\\ActivityNet\\new_val_data.json"
    val_data = ActivityNetDataset(ann_file)

    # train data num 10009
    ann_file = "G:\\ActivityNet_code\\Data2vec_MMN\\dataset\\ActivityNet\\new_train_data.json"
    train_data = ActivityNetDataset(ann_file)
