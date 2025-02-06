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

[link](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.791https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1HOPXC0/DVN/1HOPXC)

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
```
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob

wake_vision = fo.load_dataset('wake-vision-10')
wake_vision.compute_metadata()
embeddings_model = foz.load_model_zoo("mobilenet-v2-imagenet-torch")
wake_vision.compute_embeddings(model=embeddings_model, progress=True, embeddings_field="mobilenet_embeddings", skip_failures=True)
# Compute 2D Visualization using UMAP dimensionality reduction.
# pre-requisite: pip install umap-learn
results = fob.compute_visualization(wake_vision, embeddings="mobilenet_embeddings", brain_key="mobilenet_viz", method="umap", batch_size=4, num_workers=8, skip_failures=True, progress=True)
# results = fob.compute_visualization(non_labeled, embeddings="mobilenet_embeddings", brain_key="mobilenet_viz_unlabeled", method="umap", batch_size=4, num_workers=8, skip_failures=True, progress=True)
results_uniq = fob.compute_uniqueness(wake_vision, embeddings="mobilenet_embeddings", batch_size=4, num_workers=8, skip_failures=True, progress=True)
```
3-2. Compute Similiarity
3-2. Model predictions:
3-2.1 MCUNet-VWW model predictions
3-2.2 YOLO, Mobilenet, CLIP, Dino..

4- Data Exploration
4-1. Look for wrong Ground_truth & Fix it
  The approach I followed here was to run a YOLO-object detection model to get multi-labels object detections.
  Then I selected images with 'person' object detections while having a 'background' ground truth class.
  ```
  # Load YOLO model and apply it to the data
  from ultralytics imoprt YOLO
  det_yolo = YOLO("yolo11m.pt")
  wake_vision.apply_model(det_yolo, label_field="yolo_dets", batch_size=8, classes=[0], progress=True) #Applying selected classes, use GPU
  wrong_backgorund_data = wake_vision.match_labels(fields="ground_truth",filter=fo.ViewField("label")=='background').filter_labels("yolo_dets", fo.ViewField("label")=="person") # Preferably add a filter condition on person detection confidence > 0.5
  
  with fo.ProgressBar() as pb:
    for sample in pb(wrong_backgorund_data):
       sample.ground_truth.label = "person"
       sample.person = int(1)

Inversely I select images with ground_truth value as person, and then I ommit samples where yolo-detection model didn't detect a person. Most of these cases would represent a falsely labeled image.
Check screenshots from 26-01-2025 to showcase some of these mistakes


  ```
4-2. Look through unlabeled data and Select Data to be labeled (Based on Uniqueness, model error, etc..)

## Model training

### 1st experience

First, I exported the unmodified `wake-vision-10` to a Classification Directory dataset
```
dataset

└───train
│   │
│   └───person
│   |   │   file111.jpg
│   |   │   file112.jpg
│   │   │   ...
│   └───background
│   |   |   file211.jpg
|   |   |   file212.jpg
└───val
|   |   ...
|
|
└───test
|   |   ...
```
Then I used the provided docker image by doing the following changes:

```python
# Load training dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/train",
    image_size=(144, 144),  # Resize images to match model input
    batch_size=32,
    shuffle=True
)
# Load validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/val",
    image_size=(144, 144),  # Resize images to match model input
    batch_size=32,
    shuffle=True
)
# Load testing dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path + "/test",
    image_size=(144, 144),  # Resize images to match model input
    batch_size=1,
    shuffle=True
)
#some preprocessing 
data_preprocessing = tf.keras.Sequential([
    #resize images to desired input shape
    tf.keras.layers.Resizing(input_shape[0], input_shape[1])])

#try your own data augmentation recipe!
data_augmentation = tf.keras.Sequential([
    data_preprocessing,
    #apply some data augmentation 
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])

# Customization: Batching is removed from the following data loading steps not to add an extra dimension    
train_ds = train_ds.shuffle(1000).map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(lambda x, y: (data_preprocessing(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(tf.data.AUTOTUNE)
```

```bash
sudo docker run --gpus all -it --rm -v $PWD:/tmp -v $PWD/dataset:/tmp/data -w /tmp andregara/wake_vision_challenge:gpu python data_centric_track.py
```
This created a Baseline .tflite VWW_MCU_NET model to compare with, named "vww_mcu_net_baseline.tflite"

### 2nd experience

After doing data improvements with Fiftyone, I similarily export the newly obtained **labeled** dataset to a Classification Directory and re-run training with **same hyperparameters**
This creates a checkpoint "vww_mcu_net_submission.tflite"