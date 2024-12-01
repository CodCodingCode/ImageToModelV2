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

        return f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n"
