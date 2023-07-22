# Unified Static and Dynamic:Temporal Filtering Network for Efficient Spoken Video Grounding


### Video Datasets

* Download the [video feature](https://rochester.app.box.com/s/8znalh6y5e82oml2lr7to8s6ntab6mav)  provided by [2D-TAN](https://github.com/microsoft/2D-TAN)
* Download the video I3D feature of Charades-STA dataset from [LGI](https://github.com/JonghwanMun/LGI4temporalgrounding)
     ```python
      wget http://cvlab.postech.ac.kr/research/LGI/charades_data.tar.gz
      tar zxvf charades_data.tar.gz
      mv charades data
      rm charades_data.tar.gz
    ```
* Download the video C3D feature of Charades-STA dataset from [DRN](https://github.com/Alvin-Zeng/DRN)

  
### Audio Caption Datasets

* **ActivityNet Speech Dataset:** download the [original audio](https://drive.google.com/file/d/11f6sC94Swov_opNfpleTlVGyLJDFS5IW/view?usp=sharing) proposed by [VGCL](https://github.com/marmot-xy/Spoken-Video-Grounding)
* **Charades-STA Speech Dataset:** download the [original audio](https://zenodo.org/record/8019213) proposed by us.
* **TACoS Speech Dataset:** download the [original audio](https://zenodo.org/record/8022063) proposed by us.

### Dependencies

* pip install yacs h5py terminaltables tqdm librosa transformers
* conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
* conda config --add channels pytorch
* conda install pytorch-geometric -c rusty1s -c conda-forge

### Acknowledgement

