import numpy as np
import librosa
import torch
# import laion_clap
import os
from tqdm import tqdm
from transformers import AutoProcessor, Data2VecAudioModel

# quantization
# def int16_to_float32(x):
#     return (x / 32767.0).astype(np.float32)
#
#
# def float32_to_int16(x):
#     x = np.clip(x, a_min=-1., a_max=1.)
#     return (x * 32767.).astype(np.int16)
#
#
# # the sudio shape is 1*500 tensor
# def audio_clap_feat_extract(audios_path, aid_dir):
#
#     model = laion_clap.CLAP_Module(enable_fusion=False)
#     model.load_ckpt()  # download the default pretrained checkpoint.
#
#     if not os.path.exists(aid_dir):
#         os.mkdir(aid_dir)
#
#     # 遍历目标目录及其子目录
#     for root, dirs, files in os.walk(audios_path):
#         # 遍历当前目录中的所有文件
#         for file in tqdm(files):
#             audio_data, _ = librosa.load(os.path.join(root, file), sr=48000)  # sample rate should be 48000
#             audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
#             audio_data = torch.from_numpy(
#                 int16_to_float32(float32_to_int16(audio_data))).float()  # quantize before send it in to the model
#             audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
#             torch.save(audio_embed, os.path.join(aid_dir, file.split(".")[0] + '.pt'))
#
#
# # 整体保存特征内存需求较大，不建议使用
# def audio_clap_feat_all_extract(audios_path, aid_dir):
#
#     model = laion_clap.CLAP_Module(enable_fusion=False)
#     model.load_ckpt()  # download the default pretrained checkpoint.
#
#     if not os.path.exists(aid_dir):
#         os.mkdir(aid_dir)
#
#     audio_features = {}
#     # 遍历目标目录及其子目录
#     for root, dirs, files in os.walk(audios_path):
#         # 遍历当前目录中的所有文件
#         for file in tqdm(files):
#             audio_data, _ = librosa.load(os.path.join(root, file), sr=48000)  # sample rate should be 48000
#             audio_data = audio_data.reshape(1, -1)  # Make it (1,T) or (N,T)
#             audio_data = torch.from_numpy(
#                 int16_to_float32(float32_to_int16(audio_data))).float()  # quantize before send it in to the model
#             audio_embed = model.get_audio_embedding_from_data(x=audio_data, use_tensor=True)
#
#             audio_features.update(
#                 {
#                     f'{file}': audio_embed
#                 })
#     torch.save(audio_features, os.path.join(aid_dir, "audio_clap_feat.pt"))


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
            return x


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
    aid_dir = 'G:/ActivityNet_code/Explore-And-Match/data/audio_clap_feat'
    aid_dir2 = 'G:/ActivityNet_code/Explore-And-Match/data'
    aid_dir3 = 'G:\\ActivityNetDataset\\data\\audio_data2vec_feat'

    # 用clap模型提取音频特征
    # audio_clap_feat_extract(audios_path, aid_dir)
    # audio_clap_feat_all_extract(audios_path, aid_dir2)

    audio_data2vec_feat_extract(audios_path, aid_dir3)

