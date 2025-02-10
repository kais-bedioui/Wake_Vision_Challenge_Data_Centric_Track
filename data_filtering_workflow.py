import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
from fiftyone import ViewField as F

from PIL import Image
from ultralytics import YOLO

def generate_fo_dataset_from_csv(
        data_path="data",
        labels_path="data/wake_vision_train_large.csv",
        dataset_type=fo.types.CSVDataset,
    ):
    """
    This function will convert the CSV data into a FiftyOne dataset.
    Extract the image files in data_path and labels_path to 'wake_vision_train_large.csv'.
    The labels from .csv files will be added as a sample field 'person'.
    We convert that into a Label Field
    """

    wake_vision = fo.Dataset.from_dir(
        data_path=data_path,
        labels_path=labels_path,
        dataset_type=dataset_type,
        skip_missing_media=True,
        progress=True,
        include_all_data=True,
        persistent=True
    )

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
                # Non-labeled images will be tagged with 'no_label' which will be used later for auto-annotation
                sample.tags.append('no_label')
                sample.save()

    wake_vision.name = "wake-vision-data"
    wake_vision.save()
    return wake_vision

def generate_fo_metadata(dataset):
    """
    In this function we will compute samples metadata (image width, height, etc..),
    embeddings, embedding-space 2D visualisation, uniqueness scores, representativeness, similarity, and more

    These features can later be explored visually via the FiftyOne App.
    CAUTION: These methods will take time to go through a large dataset and will require significant resources.
    If the dataset size causes failures, consider applying this method on segments of the dataset
    """

    dataset.compute_metadata()
    # Here we selected Mobilenet model. Check out Fiftyone Model Zoo for more embeddings models
    embeddings_model = foz.load_model_zoo("mobilenet-v2-imagenet-torch")
    dataset.compute_embeddings(model=embeddings_model, progress=True, embeddings_field="mobilenet_embeddings", skip_failures=True)
    # Compute 2D Visualization using UMAP dimensionality reduction.
    # pre-requisite: pip install umap-learn
    r_viz = fob.compute_visualization(
        dataset,
        embeddings="mobilenet_embeddings",
        brain_key="mobilenet_viz",
        method="umap",
        batch_size=4,
        num_workers=8,
        skip_failures=True,
        progress=True
    )
    r_uniqueness = fob.compute_uniqueness(
        dataset,
        embeddings="mobilenet_embeddings",
        batch_size=4,
        num_workers=8,
        skip_failures=True,
        progress=True
    )

    r_rep = fob.compute_representativeness(
        dataset,
        embeddings="mobilenet_embeddings",
        progress=True
    )

def apply_yolo_detections(dataset, yolo_version="yolo11x.pt"):
    """
    Generates person class detections over the images
    """
    # Yolov8 model configs
    model = YOLO(yolo_version)

    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # Load image
            try:
                image = Image.open(sample.filepath)
                image_width, image_height = image.size

                # Perform inference
                # results is an object of type 
                results = model.predict(
                    source=image, 
                    imgsz=image_width,
                    conf=0.3,
                    iou=0.6,
                    classes=[0], # Detect Person class
                    device=[0], #to utilize GPU
                )
                preds = results[0]

            except Exception as e:
                filename = sample.filepath.split('/')[-1]
                print(f"Received Error {e}. -- Sample  {filename} was avoided")
                continue

            # Output example in panda format
            #         xmin        ymin        xmax        ymax     confidence  class  name
            #0    268.027100  343.181152  454.724304  389.038086    0.975919      0   boat
            #1    608.672668  341.961945  621.594055  347.252838    0.774302      0   boat

            if (preds.boxes.data.nelement() > 0):
                labels_id = preds.boxes.cls.cpu().numpy().astype('int32')
                confidence = preds.boxes.conf.cpu().numpy()
                boxes = preds.boxes.xyxyn.cpu().numpy()
                xmin = boxes[:, 0]
                ymin = boxes[:, 1]
                xmax = boxes[:, 2]
                ymax = boxes[:, 3]
                detections = []

                for x1, y1, x2, y2, label, score in zip(xmin, ymin, xmax, ymax, labels_id, confidence):
                    box = [x1, y1, x2-x1, y2-y1]
                    detections.append(fo.Detection(
                        label="person",
                        bounding_box=box,
                        confidence=score
                        )
                    )
                sample["yolo_preds"] = fo.Detections(detections=detections)
                sample.save()

def correct_images_background_to_person(dataset):
    """
    This function will use YOLO-detections to catch images that are labeled as 'background' or not-person
    but actually contain a human.

    """
    wrong_backgorund_data = (
        dataset.
        match_labels(
            fields="ground_truth", 
            filter=F("label")=='background').
            filter_labels(
                "yolo_preds",
                F("label")=="person"
            )
    )
    
    # Parse through the obtained view and correct (reverse) ground_truth info
    with fo.ProgressBar() as pb:
        for sample in pb(wrong_backgorund_data):
           sample.ground_truth.label = "person"
           sample.person = '1'
           sample.save()

    dataset.savet()

def correct_images_person_to_background(dataset):
    """
    This function will use the absence of 'person'-class YOLO-detections to catch 
    images that are labeled as 'person' but are actually background.
    """
    wrong_person_data = (
        dataset.
        match_labels(
            fields="ground_truth", 
            filter=F("label")=='person').
            filter_labels(
                "yolo_preds",
                F("label")!="person"
            )
    )
    
    # Parse through the obtained view and correct (reverse) ground_truth info
    with fo.ProgressBar() as pb:
        for sample in pb(wrong_person_data):
           sample.ground_truth.label = "background"
           sample.person = '0'
           sample.save()

    dataset.savet()

def unsupervised_labelling(dataset):
    """
    Assign ground_truth labels to unlabeled images in an usupervised manner.
    If an image contains 'person' YOLO-detection then it will be annotated as 'person'
    otherwise 'background'
    """

    # Easy method to get unlabled data (Check line 44)
    unlabeled = dataset.match_tags('no_label')
    # OR
    #unlabeled = dataset.match_labels(fields="ground_truth", filter=F() == None)
    unlabeled_person_data = unlabeled.match_labels(fields="yolo_dets", filter=(F("label") == 'person'))
    unlabeled_background_data = unlabeled.match_labels(fields="yolo_dets", filter=(F("label") != 'person')

    with fo.ProgressBar() as pb:
        for sample in pb(unlabeled_person_data):
            sample['ground_truth'] = fo.Classification(label='person')
            sample['person'] = '1'
            sample.tags.remove('no_label')
            sample.save()

    with fo.ProgressBar() as pb:
        for sample in pb(unlabeled_background_data):
            sample['ground_truth'] = fo.Classification(label='background')
            sample['person'] = '0'
            sample.tags.remove('no_label')
            sample.save()

    dataset.save()

if __name__ == "__main__":


    UNIQUENESS_THRESHOLD = 0.7
    TRAIN_DATA_SIZE = 100000
    export_dir = "/tmp/tf-wakevision-dataset"

    data_path="data",
    labels_path="data/wake_vision_train_large.csv",
    wake_vision = generate_fo_dataset_from_csv(
        data_path==data_path,
        labels_path=labels_path,
    )
    
    generate_fo_metadata(wake_vision)
    
    apply_yolo_detections(wake_vision, yolo_version="yolo11x.pt")
    
    correct_images_background_to_person(wake_vision)
    
    correct_images_person_to_background(wake_vision)
    
    unsupervised_labelling(wake_vision)

    # Select train,val, test data
    # Sort in descending order of representativeness, keep dataset with uniqueness score
    # higher than UNIQUENESS_THRESHOLD to reduce image redundancies and select
    # top TRAIN_DATA_SIZE images
    dataset = (wake_vision.
               sort_by("representativeness", reverse=True).
               match("uniqueness", F("uniqueness") >= UNIQUENESS_THRESHOLD).
               limit(TRAIN_DATA_SIZE)
    )

    test_ds = dataset[:int(0.1*len(dataset))]
    test_ds.export(
        export_dir=os.path.join(export_dir, "test")
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
        export_media=True
    )
    
    val_ds = dataset[int(0.1*len(dataset)):int(0.2*len(dataset))]
    val_ds.export(
        export_dir=os.path.join(export_dir, "val")
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
        export_media=True
    )

    train_ds = dataset[int(0.2*len(dataset)):]
    train_ds.export(
        export_dir=os.path.join(export_dir, "train")
        dataset_type=fo.types.ImageClassificationDirectoryTree,
        label_field="ground_truth",
        export_media=True
    )

    print('Dataset Created and Exported')



