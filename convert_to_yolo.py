import os


def convert_to_yolo_format(output, img_width, img_height, label):
    """
    Convert bounding box and label data to YOLO format and save to .txt file.

    Args:
        output (dict): The bounding box and label data.
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        label_dest_path (str): The destination path for the YOLO-format .txt file.
    """

    label_to_id = {
        label: 0,
        # Add other labels as needed
    }
    bboxes = output["bboxes"]
    labels = output["labels"]

    for bbox, label in zip(bboxes, labels):
        class_id = label_to_id.get(label, -1)  # Default to -1 if label not found
        if class_id == -1:
            raise ValueError(f"Label '{label}' not found in label_to_id mapping.")

        # Extract bbox values
        x_min, y_min, x_max, y_max = bbox

        # Convert to YOLO format (normalized values)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # Write to file in YOLO format

        return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
