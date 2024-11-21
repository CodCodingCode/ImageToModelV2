import os

# Map labels to class IDs (you can modify this as needed)
label_to_id = {
    "car": 0,
    # Add other labels as needed
}


def convert_to_yolo_format(output, img_width, img_height, output_dir, img_name):
    """
    Convert bounding box and label data to YOLO format and save to .txt file.

    Args:
        output (dict): The bounding box and label data.
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        output_dir (str): Directory to save the YOLO-format .txt file.
        img_name (str): Name of the image (without extension).
    """
    bboxes = output["bboxes"]
    labels = output["labels"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Prepare the .txt file path
    txt_file_path = os.path.join(output_dir, f"{img_name}.txt")

    with open(txt_file_path, "w") as f:
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
            f.write(
                f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
            )


# Example usage
output = {"bboxes": [[34.24, 160.08, 597.44, 371.76]], "labels": ["car"]}
img_width = 640  # Example image width
img_height = 480  # Example image height
output_dir = "yolo_labels"  # Directory to save labels
img_name = "image1"  # Image name without extension

convert_to_yolo_format(output, img_width, img_height, output_dir, img_name)
