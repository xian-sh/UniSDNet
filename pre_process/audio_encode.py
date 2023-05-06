import numpy as np
import librosa
import torch
import os
from tqdm import tqdm
from transformers import AutoProcessor, Data2VecAudioModel

class Data2vecModel(torch.nn.Module):
    def __init__(self):
        super(Data2vecModel, self).__init__()

        self.model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
        self.processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
        self.norm = torch.nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, data, sr):
        inputs = self.processor(data, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            x = self.model(**inputs)
            x = x.last_hidden_state
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.transpose(1, 2)
            return x  # (1,1,678)


# data2vec
def audio_data2vec_feat_extract(audios_path, aid_dir):

    audio_model = Data2vecModel()

    if not os.path.exists(aid_dir):
        os.mkdir(aid_dir)

    # 遍历目标目录及其子目录
    for root, dirs, files in os.walk(audios_path):
        # 遍历当前目录中的所有文件
        for file in tqdm(files):
            save_path = os.path.join(aid_dir, file.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            audio_data, _ = librosa.load(os.path.join(root, file), sr=16000)  # sample rate should be 16000
            outputs = audio_model(audio_data, sr=16000)

            torch.save(outputs, save_path)


if __name__ == '__main__':
    audios_path = 'H:/Code/SVG/data/wavs_16k'
    aid_dir = r'.\data\audio_data2vec_feat'  # save path

    audio_data2vec_feat_extract(audios_path, aid_dir)

