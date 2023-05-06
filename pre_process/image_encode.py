import os
import sys
import glob
import json
import argparse
import torch
import torch.nn as nn
import lib.modeling.models.clip as clip
import time
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
sys.path.append(os.path.dirname(os.path.abspath(
                os.path.dirname((__file__)))))
from lib.modeling.models.model_clip import build_image_clip
from lib.utils.misc import AverageMeter
from transformers import AutoImageProcessor, Data2VecVisionModel


def encode_image_with_clip(dir_to_data, num_frames):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_meters = defaultdict(AverageMeter)
    tictoc = time.time()

    clip_model = 'ViT-B/32'  # variants of clip model choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14']
    # load clip model
    model, preprocess = clip.load(clip_model, device=device)
    model_image = build_image_clip(model)
    time_meters['load_model'].update(time.time()-tictoc)
    tictoc = time.time()

    dir_to_anno = '../data/activitynet/annotations'

    phases = ['train', 'val']
    for phase in phases:
        # load annotations
        with open(os.path.join(dir_to_anno, phase + '.json')) as j:
            annos = json.load(j, encoding='utf-8')

        time_meters['load_annotations'].update(time.time()-tictoc)
        tictoc = time.time()

        for video_id in tqdm(list(annos.keys()), desc=phase):
            save_dir = os.path.join(dir_to_data, 'clip_features', phase, video_id, )
            if os.path.exists(save_dir):
                if os.path.exists(os.path.join(save_dir, f'vid_feats_{str(num_frames)}.pt')):
                    continue
            else:
                os.makedirs(save_dir)

            dir_to_frame = os.path.join(dir_to_data, 'frames', str(num_frames), video_id+'*')

            if not os.path.exists(dir_to_frame):
                print(f'The directory {dir_to_frame} does not exists.')

            frames = sorted(glob.glob(os.path.join(dir_to_frame, '*.png')))

            if len(frames) == 0:
                print(f'No valid frames exist in {dir_to_frame}.')
                continue
            video_frames = [Image.open(frame).convert('RGB') for frame in frames]
            time_meters['prepare_image'].update(time.time()-tictoc)
            tictoc = time.time()

            # preprocess
            frames = torch.cat([preprocess(video_frame).unsqueeze(0).to(device) for video_frame in video_frames], dim=0)
            time_meters['preprocess_image'].update(time.time()-tictoc)
            tictoc = time.time()

            # encode
            with torch.no_grad():
                video_features = model_image(frames)  # Nx512
            time_meters['encode_image'].update(time.time()-tictoc)
            tictoc = time.time()

            torch.save(video_features.cpu(), os.path.join(save_dir, 'vid_feats_' + str(num_frames))+'.pt')
            time_meters['save_features'].update(time.time()-tictoc)
            tictoc = time.time()

    print('Time stats:')
    for name, meter in time_meters.items():
        d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
        print(f'{name} ==> {d}')


# data2vec
class VideoModel(nn.Module):
    def __init__(self):
        super(VideoModel, self).__init__()

        # self.processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
        self.model = Data2VecVisionModel.from_pretrained("facebook/data2vec-vision-base")
        self.norm = nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, data):
        # inputs = self.processor(data, return_tensors="pt")
        with torch.no_grad():
            x = self.model(**data)
            x = x.last_hidden_state
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.transpose(1, 2)
        return x


def image_data2vec_feat_extract(dir_to_data, num_frames):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    time_meters = defaultdict(AverageMeter)
    tictoc = time.time()

    processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    model_image = VideoModel().to(device)
    time_meters['load_model'].update(time.time() - tictoc)
    tictoc = time.time()

    videos = os.path.join(dir_to_data, 'frames', str(num_frames))
    videos_list = os.listdir(videos)

    for video_id in tqdm(videos_list):
        save_dir = os.path.join(dir_to_data, 'data2vec_features', video_id)

        if os.path.exists(save_dir):
            if os.path.exists(os.path.join(save_dir, f'vid_feats_{str(num_frames)}.pt')):
                continue
        else:
            os.makedirs(save_dir)

        dir_to_frame = os.path.join(dir_to_data, 'frames', str(num_frames), video_id)

        if not os.path.exists(dir_to_frame):
            print(f'The directory {dir_to_frame} does not exists.')

        frames = sorted(glob.glob(os.path.join(dir_to_frame, '*.png')))

        if len(frames) == 0:
            print(f'No valid frames exist in {dir_to_frame}.')
            continue
        frames = [Image.open(frame).convert('RGB') for frame in frames]
        time_meters['prepare_image'].update(time.time() - tictoc)
        tictoc = time.time()

        frames = processor(frames, return_tensors="pt").to(device)

        # encode
        with torch.no_grad():
            video_features = model_image(frames).squeeze()  # Nx768
        time_meters['encode_image'].update(time.time() - tictoc)
        tictoc = time.time()

        torch.save(video_features.cpu(), os.path.join(save_dir, 'vid_feats_' + str(num_frames))+'.pt')
        time_meters['save_features'].update(time.time() - tictoc)
        tictoc = time.time()

    print('Time stats:')
    for name, meter in time_meters.items():
        d = {k: f'{getattr(meter, k):.4f}' for k in ['max', 'min', 'avg']}
        print(f'{name} ==> {d}')


def single_image_data2vec_feat_extract(dir_to_data, video_id):

    processor = AutoImageProcessor.from_pretrained("facebook/data2vec-vision-base")
    model_image = VideoModel()

    dir_to_frames = os.path.join(dir_to_data, 'frames', str(num_frames), video_id)
    frames = sorted(glob.glob(os.path.join(dir_to_frames, '*.png')))

    save_dir = os.path.join(dir_to_data, 'data2vec_features', video_id)
    # encode
    with torch.no_grad():
        video_features = model_image(frames).squeeze()  # Nx768
    torch.save(video_features.cpu(), os.path.join(save_dir, 'vid_feats_' + str(num_frames)) + '.pt')


if __name__ == '__main__':

    num_frames = 64  # number of input frames
    image_only = False  # if image_only == True, only image features are extracted.
    dir_to_data = 'G:\\ActivityNet_code\\Explore-And-Match\\data\\vedio\\activitynet'

    # encode_image_with_clip(dir_to_data, num_frames)

    image_data2vec_feat_extract(dir_to_data, num_frames)

    # G:\ActivityNet_code\Explore-And-Match\data\vedio\activitynet\frames\64\v_0fw8it7Gj7k.mp4.
    single_image_data2vec_feat_extract(dir_to_data, 'v_0fw8it7Gj7k.mp4')