# 🚀 **Data-Centric Track**

This is my submission for the Data-Centric challenge

# Pre-requisites

## Setup environment

1- Create conda environment `conda create -n wakevision python=3.9`
2- 
```
conda activate wakevision
pip install fiftyone
# create a 'dataset' folder and download the needed chunks of data from
# [link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.791https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1HOPXC0/DVN/1HOPXC)
# In my example I dowloaded wake-vision-10 (~7GB) + wake_vision_train_large.csv
# Change column name in the .csv file to 'filepath'
```

2 - Create A FiftyOne Dataset from the csv data and assign ground_truth label field to each sample.  
```
# python3
import fiftyone as fo
wake_vision = fo.Dataset.from_dir(data_path="data", labels_path="data/wake_vision_train_large.csv", dataset_type=fo.types.CSVDataset, skip_missing_media=True, progress=True, include_all_data=True, persistent=True)
# Create a Label Field for Ground Truth
# Images with no GT are given a sample tag 'no_label'
with fo.ProgressBar() as pb:
    for sample in pb(wake_vision):
        if sample.person:
            if int(sample.person)==1:
                sample["ground_truth"] = fo.Classification(label="person")
                sample.save()
                continue
            elif int(sample.person)==0:
                sample["ground_truth"] = fo.Classification(label="background")
                sample.save()
                continue
        else:
            sample.tags.append('no_label')
            sample.save()

session = fo.launch_app(wake_vision)
```

3- Data & Metadata
3-1. Compute embeddings with Mobilenet  & Visualization
3-2. Compute Similiarity
3-2. Model predictions:
3-2.1 MCUNet-VWW model predictions
3-2.2 YOLO, Mobilenet, CLIP, Dino..

4- Data Exploration
4-1. Look for wrong Ground_truth & Fix it with CVAT
4-2. Look through unlabeled data and Select Data to be labeled (Based on Uniqueness, model error, etc..)

## Model training
