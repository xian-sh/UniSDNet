import os
import json
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch

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


text_path1 = r'path\TACoS\train_audio_new.json'
text_path2 = r'path\TACoS\test_audio_new.json'
text_path3 = r'path\TACoS\val_audio_new.json'

text_dir = r'path\TACoS\text_glove_feat_new'

text_glove_feat_extract(text_path1,text_dir)
text_glove_feat_extract(text_path2,text_dir)
text_glove_feat_extract(text_path3,text_dir)
