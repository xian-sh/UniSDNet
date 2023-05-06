## Step1: Download ActivityNet dataset original videos:

the full dataset available on Google and Baidu drives. Please fill in this [request form](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform) to have a 7-day-access to download the videos from the drive folders. 

more details look at the link:http://activity-net.org/download.html

what files are you need?

```bash
source_dir/
    missing_files.zip
    missing_files_v1-2_test.zip
    missing_files_v1-3_test.zip
    v1-2_test.tar.gz
    v1-2_train.tar.gz
    v1-2_val.tar.gz
    v1-3_test.tar.gz
    v1-3_train_val.tar.gz
```

you can also get the v1-2 and v1-3 data through the baidu disk link I shared with you: 

https://pan.baidu.com/s/1LUcCE5vQoZEcenaBqExKwA?pwd=mo4k

### Or you can download ActivityNet dataset's C3D feature data from: 

https://cs.stanford.edu/people/ranjaykrishna/densevid/

## Step2: Download the caption data:

the original split(no audio):   https://cs.stanford.edu/people/ranjaykrishna/densevid/
    
    or look at the 'root/dataset/ActivityNet/' folder, which includes '{train/val/test}.json'


caption file include audio(**record per caption individually**): 

    look at the 'root/dataset/ActivityNet/' folder, which includes 'new_{train/val/test}_data.json'


caption file include audio(**all record of video related captions**):
   
    look at the 'root/dataset/ActivityNet/' folder, which includes '{train/val/test}_audio.json'

## Step3: Split data(if you want to do the work by starting with feature extraction from video):

Note that above caption files are correspond to Anet1.3(activitynet-200),so if you want to get the whole videos split corresponding to the captions, you should execute the following code. 

```bash
pip install fiftyone
```


```python
import fiftyone as fo
import fiftyone.zoo as foz

source_dir = "/path/to/dir-with-activitynet-files"

# make a new dir to store the dataset
download_dir = r'/path/to/save/activitynet-200'

# use a disk with more space to store the dataset, default is C:\Users\user\fiftyone
fo.config.dataset_zoo_dir = download_dir  
fo.config.default_dataset_dir = download_dir
print(fo.config)

# Load the entire ActivityNet 200 dataset into FiftyOne
dataset = foz.load_zoo_dataset(r"activitynet-200", source_dir=source_dir)

session = fo.launch_app(dataset)
```

where source_dir contains the source files in the following format(**these files are from your the download in step1**):

```bash
source_dir/
    missing_files.zip
    missing_files_v1-2_test.zip
    missing_files_v1-3_test.zip
    v1-2_test.tar.gz
    v1-2_train.tar.gz
    v1-2_val.tar.gz
    v1-3_test.tar.gz
    v1-3_train_val.tar.gz
```
If you have already decompressed the archives, that is okay too:

```bash
source_dir/
    missing_files/
        v_<id>.<ext>
        ...
    missing_files_v1-2_test/
        v_<id>.<ext>
        ...
    missing_files_v_1-3_test/
        v_<id>.<ext>
        ...
    v1-2/
        train/
            v_<id>.<ext>
            ...
        val/
            ...
        test/
            ...
    v1-3/
        train_val/
            v_<id>.<ext>
            ...
        test/
            ...
```

fiftyone tools reference: https://docs.voxel51.com/integrations/activitynet.html#activitynet-full-split-downloads
