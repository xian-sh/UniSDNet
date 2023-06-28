# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
import soundfile as sf
from datasets import load_dataset
import numpy as np
import os
from tqdm import tqdm
import json

# set random seed for reproducibility
np.random.seed(1)
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split='validation')


def text_microsoft_mav(root_path, text_path, audios_dir):

    if not os.path.exists(audios_dir):
        os.mkdir(audios_dir)

    if 'train' in text_path:
        mode = 'train'
    elif 'val' in text_path:
        mode = 'val'
    elif 'test' in text_path:
        mode = 'test'
    else:
        raise ValueError("text_path should contain train, val or test")

    audio_json = {}
    audio_json_path = os.path.join(root_path, f"{mode}_audio.json")

    data = json.load(open(text_path, 'r', encoding='utf-8'))

    # generate random speaker id for each video
    random_idx = np.random.randint(0, len(embeddings_dataset), size=len(data))

    for i, (k, v) in tqdm(enumerate(data.items()), total=len(data)):
        vid = k
        duration = v['duration']
        timestamps = v['timestamps']
        sentences = v['sentences']
        audios = []
        speaker = embeddings_dataset[int(random_idx[i])]["filename"]
        speaker_id = int(random_idx[i])
        for i, s in enumerate(sentences):
            audio_name = f"{vid}_{mode}_{i+1}.wav"
            save_path = os.path.join(audios_dir, audio_name)
            audios.append(audio_name)
            if os.path.exists(audio_name):
                continue
            else:
                inputs = processor(text=s, return_tensors="pt")
                speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                sf.write(save_path, speech.numpy(), samplerate=16000)

        audio_json.update({
            f"{vid}": {
                'duration': duration,
                'timestamps': timestamps,
                'sentences': sentences,
                'audios': audios,
                'speaker': speaker,
                'speaker_id': speaker_id,
                }
            })

    json.dump(audio_json, open(audio_json_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


def TACoS_text_microsoft_mav(root_path, text_path, audios_dir):

    if not os.path.exists(audios_dir):
        os.mkdir(audios_dir)

    if 'train' in text_path:
        mode = 'train'
    elif 'val' in text_path:
        mode = 'val'
    elif 'test' in text_path:
        mode = 'test'
    else:
        raise ValueError("text_path should contain train, val or test")

    audio_json = {}
    audio_json_path = os.path.join(root_path, f"{mode}_audio.json")

    data = json.load(open(text_path, 'r', encoding='utf-8'))

    # generate random speaker id for each video
    random_idx = np.random.randint(0, len(embeddings_dataset), size=len(data))

    for i, (k, v) in tqdm(enumerate(data.items()), total=len(data)):
        vid = k
        vid_name = vid.split(".")[0]
        num_frames = v['num_frames']
        fps = v['fps']
        timestamps = v['timestamps']
        sentences = v['sentences']
        audios = []
        speaker = embeddings_dataset[int(random_idx[i])]["filename"]
        speaker_id = int(random_idx[i])
        for i, s in enumerate(sentences):
            audio_name = f"{vid_name}_{mode}_{i+1}.wav"
            save_path = os.path.join(audios_dir, audio_name)
            audios.append(audio_name)
            if os.path.exists(audio_name):
                continue
            else:
                inputs = processor(text=s, return_tensors="pt")
                speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]["xvector"]).unsqueeze(0)
                speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                sf.write(save_path, speech.numpy(), samplerate=16000)

        audio_json.update({
            f"{vid}": {
                'num_frames': num_frames,
                'fps': fps,
                'timestamps': timestamps,
                'sentences': sentences,
                'audios': audios,
                'speaker': speaker,
                'speaker_id': speaker_id,
                }
            })

    json.dump(audio_json, open(audio_json_path, 'w', encoding='utf-8'), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    # root_path = r"G:\ActivityNet_code\Data2vec_MMN\dataset\Charades_STA"
    root_path = r"G:\ActivityNet_code\Data2vec_MMN\dataset\TACoS"
    train_path = os.path.join(root_path, "train.json")
    val_path = os.path.join(root_path, "val.json")
    test_path = os.path.join(root_path, "test.json")

    audios_dir = r"G:\ActivityNetDataset\data\TACoS\audios"

    # Charades_STA: train:5336  test:1334
    # TACoS: train:75  val:27    test: 25
    TACoS_text_microsoft_mav(root_path, test_path, audios_dir)
    TACoS_text_microsoft_mav(root_path, val_path, audios_dir)
    TACoS_text_microsoft_mav(root_path, train_path, audios_dir)