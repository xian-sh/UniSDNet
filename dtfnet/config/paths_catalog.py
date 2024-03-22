"""Centralized catalog of paths."""
import os
from dtfnet.config import cfg

class DatasetCatalog(object):
    DATA_DIR = ""

    DATASETS = {
        "tacos_train":{
            "audio_dir": "/hujingjing2/data/TACoS/audio_data2vec_feat",        # text or audio feature path
            "ann_file": "./dataset/TACoS/train_audio_new.json",
            "feat_file": "/hujingjing2/data/TACoS/tall_c3d_features.hdf5",     # video feature path
        },
        "tacos_val":{
            "audio_dir": "/hujingjing2/data/TACoS/audio_data2vec_feat",
            "ann_file": "./dataset/TACoS/val_audio_new.json",
            "feat_file": "/hujingjing2/data/TACoS/tall_c3d_features.hdf5",
        },
        "tacos_test":{
            "audio_dir": "/hujingjing2/data/TACoS/audio_data2vec_feat",
            "ann_file": "./dataset/TACoS/test_audio_new.json",
            "feat_file": "/hujingjing2/data/TACoS/tall_c3d_features.hdf5",
        },
        "activitynet_train":{
            "audio_dir": "/hujingjing2/data/ActivityNet/text_distbert_feat",
            "ann_file": "./dataset/ActivityNet/train_audio_new.json",
            "feat_file": "/hujingjing2/Spoken_Video_Grounding/data/activity-c3d",
        },
        "activitynet_val":{ 
            "audio_dir": "/hujingjing2/data/ActivityNet/text_distbert_feat",   #/hujingjing2/data/ActivityNet/text_distbert_feat         
            "ann_file": "./dataset/ActivityNet/val_audio_new.json",
            "feat_file": "/hujingjing2/Spoken_Video_Grounding/data/activity-c3d",
        },
        "activitynet_test":{
            "audio_dir": "/hujingjing2/data/ActivityNet/text_distbert_feat",
            "ann_file": "./dataset/ActivityNet/test_audio_new.json",
            "feat_file": "/hujingjing2/Spoken_Video_Grounding/data/activity-c3d",
        },
        "charades_train": {
            "audio_dir": "/hujingjing2/data/Charades_STA/text_distbert_feat",
            "ann_file": "./dataset/Charades_STA/train_audio_new.json",
            "feat_file": "/hujingjing2/data/data/features/i3d_finetuned",
            
        },
        "charades_test": {
            "audio_dir": "/hujingjing2/data/Charades_STA/text_distbert_feat",
            "ann_file": "./dataset/Charades_STA/test_audio_new.json",
             "feat_file": "/hujingjing2/data/data/features/i3d_finetuned",
            
        },
    }

    @staticmethod
    def get(name):
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]
        
        if "charades" in name and cfg.MODEL.DTF.VIDEO_MODE == 'vgg':
            attrs["feat_file"] = "/hujingjing2/data/Charades_STA/vgg_rgb_features.hdf5"     # change this path with your own charades video feature path
        if "charades" in name and cfg.MODEL.DTF.VIDEO_MODE == 'c3d':
            attrs["feat_file"] = "/hujingjing2/data/Charades_STA/C3D_unit16_overlap0.5"
            
        if "charades" in name and cfg.MODEL.DTF.VIDEO_MODE == 'c3d_pca':
            attrs["feat_file"] = "/hujingjing2/data/Charades_STA/C3D_PCA_new"  
        
        if "charades" in name and cfg.MODEL.DTF.VIDEO_MODE == 'i3d':
            attrs["feat_file"] = "/hujingjing2/data/data/features/i3d_finetuned"
            
        args = dict(
            audio_dir=os.path.join(data_dir, attrs["audio_dir"]),
            ann_file=os.path.join(data_dir, attrs["ann_file"]),
            feat_file=os.path.join(data_dir, attrs["feat_file"]),
        )
        if "tacos" in name:
            return dict(
                factory="TACoSDataset",
                args=args,
            )
        elif "activitynet" in name:
            return dict(
                factory = "ActivityNetDataset",
                args = args
            )
        elif "charades" in name:
            return dict(
                factory = "CharadesDataset",
                args = args
            )
        raise RuntimeError("Dataset not available: {}".format(name))

