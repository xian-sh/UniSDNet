'''
Note the dependency package torch for sentence_transformers is version 2.0.1, 
it is recommended to create a new Conda environment and then
pip install sentence_transformers
'''

import os
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, Data2VecTextModel, DistilBertModel

def text_glove_feat_extract(text_path, text_dir):
    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')  # 'sentence-transformers/average_word_embeddings_glove.840B.300d'
    if not os.path.exists(text_dir):
        os.mkdir(text_dir)

    data = json.load(open(text_path, 'r', encoding='utf-8'))

    for value in tqdm(data.values()):
        sent = value['sentences']
        aid_name = value['audios']
        for s, a in zip(sent, aid_name):
            save_path = os.path.join(text_dir, a.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            outputs = model.encode(s, convert_to_numpy=False).unsqueeze(0)
            torch.save(outputs, save_path)

class Data2vecModel(torch.nn.Module):
    def __init__(self):
        super(Data2vecModel, self).__init__()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/data2vec-text-base")
        self.model = Data2VecTextModel.from_pretrained("facebook/data2vec-text-base")

        self.norm = torch.nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, data):
        inputs = self.tokenizer(data, return_tensors="pt")
        with torch.no_grad():
            x = self.model(**inputs)
            x = x.last_hidden_state
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.transpose(1, 2).squeeze(1)
            # print(x.shape)
            return x


# data2vec
def text_data2vec_feat_extract(text_path, text_dir):

    text_model = Data2vecModel()

    if not os.path.exists(text_dir):
        os.mkdir(text_dir)

    data = json.load(open(text_path, 'r', encoding='utf-8'))

    for value in tqdm(data.values()):
        sent = value['sentences']
        aid_name = value['audios']
        for s, a in zip(sent, aid_name):
            save_path = os.path.join(text_dir, a.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            outputs = text_model(s)
            torch.save(outputs, save_path)


class DistilBert(torch.nn.Module):
    def __init__(self):
        super(DistilBert, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")

        self.norm = torch.nn.LayerNorm(768, eps=1e-05, elementwise_affine=True)
        self.avgpool = torch.nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, data):
        inputs = self.tokenizer(data, return_tensors="pt")
        with torch.no_grad():
            x = self.model(**inputs)
            x = x.last_hidden_state
            x = self.norm(x)
            x = self.avgpool(x.transpose(1, 2))
            x = x.transpose(1, 2).squeeze(1)
            # print(x.shape)
            return x


def text_distbert_feat_extract(text_path, text_dir):

    text_model = DistilBert()

    if not os.path.exists(text_dir):
        os.mkdir(text_dir)

    data = json.load(open(text_path, 'r', encoding='utf-8'))

    for value in tqdm(data.values()):
        sent = value['sentences']
        aid_name = value['audios']
        for s, a in zip(sent, aid_name):
            save_path = os.path.join(text_dir, a.split(".")[0] + '.pt')
            if os.path.exists(save_path):
                continue

            outputs = text_model(s)
            torch.save(outputs, save_path)


if __name__ == '__main__':            
    text_path1 = r'path\TACoS\train_audio_new.json'
    text_path2 = r'path\TACoS\test_audio_new.json'
    text_path3 = r'path\TACoS\val_audio_new.json'

    text_dir1 = r'path\TACoS\text_glove_feat_new'
    text_dir2 = r'path\TACoS\text_distbert_feat_new'

    text_glove_feat_extract(text_path1,text_dir1)
    text_glove_feat_extract(text_path2,text_dir1)
    text_glove_feat_extract(text_path3,text_dir1)

    text_distbert_feat_extract(text_path1,text_dir2)
    text_distbert_feat_extract(text_path2,text_dir2)
    text_distbert_feat_extract(text_path3,text_dir2)
