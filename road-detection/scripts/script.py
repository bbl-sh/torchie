import os
import cv2

def parse_yolo_annotation(image_dir, annotation_dir, class_labels):
    """
    Parse YOLO annotation files and prepare data for training.
    Args:
        image_dir (str): Path to directory containing images.
        annotation_dir (str): Path to directory containing YOLO annotation files.
        class_labels (list): List of class labels (strings).
    Returns:
        list: A list of dictionaries with 'image', 'boxes', and 'labels'.
    """
    data = []

    for filename in os.listdir(annotation_dir):
        if filename.endswith(".txt"):
            # Get corresponding image file
            image_file = filename.replace(".txt", ".jpg")
            image_path = os.path.join(image_dir, image_file)
            annotation_path = os.path.join(annotation_dir, filename)

            # Read image to get dimensions
            image = cv2.imread(image_path)
            height, width, _ = image.shape

            boxes = []
            labels = []

            # Read YOLO annotation file
            with open(annotation_path, "r") as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    class_id = int(class_id)

                    # Convert YOLO format to absolute pixel values
                    x_min = int((x_center - w / 2) * width)
                    y_min = int((y_center - h / 2) * height)
                    x_max = int((x_center + w / 2) * width)
                    y_max = int((y_center + h / 2) * height)

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id)

            data.append({
                "image": image_file,
                "boxes": boxes,
                "labels": labels
            })

    return data
