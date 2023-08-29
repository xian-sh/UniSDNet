import librosa
import torch
import os
from tqdm import tqdm
from transformers import AutoProcessor, Wav2Vec2Model, Data2VecAudioModel, ASTModel, UniSpeechSatModel


class Wav2vec2Model(torch.nn.Module):
    def __init__(self):
        super(Wav2vec2Model, self).__init__()

        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        self.norm = torch.nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, data, sr):
        inputs = self.processor(data, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            x = self.model(**inputs)
            x = x.last_hidden_state
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.transpose(1, 2).squeeze(0)  # (1, 1, 768) -> (1, 768)
            # print(x.shape)
            return x


# wav2vec2
def audio_wav2vec2_feat_extract(audios_path, aid_dir):

    audio_model = Wav2vec2Model()

    if not os.path.exists(aid_dir):
        os.mkdir(aid_dir)

    file_list = os.listdir(audios_path)
    for file in tqdm(file_list):
        save_path = os.path.join(aid_dir, file.split(".")[0] + '.pt')
        # print(save_path)
        if os.path.exists(save_path):
            continue

        audio_data, _ = librosa.load(os.path.join(audios_path, file), sr=16000)  # sample rate should be 16000
        # print(audio_data.shape)
        outputs = audio_model(audio_data, sr=16000)

        torch.save(outputs, save_path)


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
            x = x.transpose(1, 2).squeeze(0)  # (1, 1, 768) -> (1, 768)
            # print(x.shape)
            return x


# data2vec
def audio_data2vec_feat_extract(audios_path, aid_dir):

    audio_model = Data2vecModel()

    if not os.path.exists(aid_dir):
        os.mkdir(aid_dir)

    file_list = os.listdir(audios_path)
    for file in tqdm(file_list):
        save_path = os.path.join(aid_dir, file.split(".")[0] + '.pt')
        # print(save_path)
        if os.path.exists(save_path):
            continue

        audio_data, _ = librosa.load(os.path.join(audios_path, file), sr=16000)  # sample rate should be 16000
        # print(audio_data.shape)
        outputs = audio_model(audio_data, sr=16000)

        torch.save(outputs, save_path)


class ASTaudioModel(torch.nn.Module):
    def __init__(self):
        super(ASTaudioModel, self).__init__()

        self.model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        self.processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
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

# ast
def audio_ast_feat_extract(audios_path, aid_dir):

    audio_model = ASTaudioModel()
    if not os.path.exists(aid_dir):
        os.mkdir(aid_dir)

    for root, dirs, files in os.walk(audios_path):
        for file in tqdm(files):
            save_path = os.path.join(aid_dir, file.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            audio_data, _ = librosa.load(os.path.join(root, file), sr=16000)  # sample rate should be 16000
            outputs = audio_model(audio_data, sr=16000)

            torch.save(outputs, save_path)

class UniSpeechModel(torch.nn.Module):
    def __init__(self):
        super(UniSpeechModel, self).__init__()

        self.model = UniSpeechSatModel.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
        self.processor = AutoProcessor.from_pretrained("microsoft/unispeech-sat-base-100h-libri-ft")
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

# unispeech
def audio_unispeech_feat_extract(audios_path, aid_dir):

    audio_model = UniSpeechModel()
    if not os.path.exists(aid_dir):
        os.mkdir(aid_dir)
    for root, dirs, files in os.walk(audios_path):
        for file in tqdm(files):
            save_path = os.path.join(aid_dir, file.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            audio_data, _ = librosa.load(os.path.join(root, file), sr=16000)  # sample rate should be 16000
            outputs = audio_model(audio_data, sr=16000)

            torch.save(outputs, save_path)


if __name__ == '__main__':
    audios_path = 'H:/Code/SVG/data/wavs_16k'

    aid_dir = r'path\TACoS\audio_data2vec_feat'
    aid_dir1 = r'path\TACoS\audio_wav2vec2_feat'
    audio_data2vec_feat_extract(audios_path, aid_dir)
    audio_wav2vec2_feat_extract(audios_path, aid_dir1)
