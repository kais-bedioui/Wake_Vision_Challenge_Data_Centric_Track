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
    _ = fob.compute_visualization(
        dataset,
        embeddings="mobilenet_embeddings",
        brain_key="mobilenet_viz",
        method="umap",
        batch_size=4,
        num_workers=8,
        skip_failures=True,
        progress=True
    )
    _ = fob.compute_uniqueness(
        dataset, 
        embeddings="mobilenet_embeddings", 
        batch_size=4, 
        num_workers=8, 
        skip_failures=True, 
        progress=True
    )

    _ = fob.compute_representativeness(
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

def correct_false_background_images(dataset):
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
           sample.person = int(1)


if __name__ == "__main__":

    data_path="data",
    labels_path="data/wake_vision_train_large.csv",
    wake_vision = generate_fo_dataset_from_csv(
        data_path==data_path,
        labels_path=labels_path,
    )
    generate_fo_metadata(wake_vision)
    apply_yolo_detections(wake_vision, yolo_version="yolo11x.pt")