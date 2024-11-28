import os
import shutil
import yaml
from math import floor
from random import shuffle
from convert_to_yolo import convert_to_yolo_format
from create_data import run_example, convert_to_od_format
from ultralytics import YOLO


def generate_label_for_image(image_path, task_prompt, text_input, label):

    results, width, height = run_example(
        task_prompt=task_prompt, image=image_path, text_input=text_input
    )
    data = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])

    message = convert_to_yolo_format(
        data, img_width=width, img_height=height, label=label
    )
    return message


def split_dataset_with_labels(
    input_dir, train_ratio, test_ratio, valid_ratio, input_class
):
    """
    Splits images and their corresponding labels into train, test, and validation sets.

    Args:
        input_dir (str): Path to the input folder containing images.
        train_ratio (float): Proportion of images for training.
        test_ratio (float): Proportion of images for testing.
        valid_ratio (float): Proportion of images for validation.
    """
    # Validate ratios
    if not abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6:
        raise ValueError("The sum of train, test, and valid ratios must equal 1.")

    # List all image files
    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
    }  # Add valid image extensions
    images = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f))
        and os.path.splitext(f)[1].lower() in valid_extensions
    ]
    shuffle(images)  # Shuffle images for randomness

    # Calculate split indices
    total_images = len(images)
    train_count = floor(total_images * train_ratio)
    test_count = floor(total_images * test_ratio)
    valid_count = total_images - train_count - test_count

    train_images = images[:train_count]
    test_images = images[train_count : train_count + test_count]
    valid_images = images[train_count + test_count :]

    # Define static output directory
    output_dir = "output"
    for split in ["train", "test", "valid"]:
        os.makedirs(os.path.join(output_dir, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, "labels"), exist_ok=True)

    # Copy images and generate labels for each split
    def process_files(file_list, split_name):
        for file_name in file_list:
            image_src_path = os.path.join(input_dir, file_name)
            
            # Check if the file is a valid image
            if not os.path.isfile(image_src_path):
                print(f"File not found: {image_src_path}")
                continue
            
            try:
                label_data = generate_label_for_image(
                    image_src_path, "<OPEN_VOCABULARY_DETECTION>", input_class, input_class
                )
            except Exception as e:
                print(f"Error processing {image_src_path}: {e}")
                continue

            # Save image
            image_dest_path = os.path.join(output_dir, split_name, "images", file_name)
            shutil.copy(image_src_path, image_dest_path)

            # Save label (assuming label filename matches image filename but with .txt extension)
            label_filename = os.path.splitext(file_name)[0] + ".txt"
            label_dest_path = os.path.join(
                output_dir, split_name, "labels", label_filename
            )
            with open(label_dest_path, "w") as label_file:
                label_file.write(label_data)

    process_files(train_images, "train")
    process_files(test_images, "test")
    process_files(valid_images, "valid")

    print(f"Dataset split completed:")
    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(test_images)} images")
    print(f"Valid: {len(valid_images)} images")
    print(f"Output saved in 'output' folder.")


def create_yaml_file(output_dir, class_names, yaml_file_path="dataset.yaml"):
    """
    Creates a .yaml file for YOLO training.

    Args:
        output_dir (str): The output directory containing 'train' and 'valid' splits.
        class_names (list): List of class names for the dataset.
        yaml_file_path (str): Path to save the .yaml file.
    """
    train_images_path = os.path.abspath(os.path.join(output_dir, "train", "images"))
    valid_images_path = os.path.abspath(os.path.join(output_dir, "valid", "images"))
    test_images_path = os.path.abspath(os.path.join(output_dir, "test", "images"))

    # Construct the YAML content
    data = {
        "train": train_images_path,
        "val": valid_images_path,
        "test": test_images_path,
        "nc": 1,  # Number of classes
        "names": [class_names],  # Class names
    }

    # Write the YAML file
    with open(yaml_file_path, "w") as yaml_file:
        yaml.dump(data, yaml_file, default_flow_style=False)

    print(f"YAML file created at: {yaml_file_path}")


# Example Usage
output_dir = "output"  # Static output directory where train/valid splits are saved
class_names = [
    "car"
]  # Update this list with your classes (e.g., ['car', 'person', ...])
yaml_file_path = "dataset.yaml"  # Path to save the .yaml file

create_yaml_file(output_dir, class_names, yaml_file_path)

# User inputs
input_dir = input("Enter the path to the folder containing images: ").strip()

train_ratio = float(input("Enter the train split ratio (e.g., 0.7 for 70%): ").strip())
test_ratio = float(input("Enter the test split ratio (e.g., 0.2 for 20%): ").strip())
valid_ratio = float(
    input("Enter the validation split ratio (e.g., 0.1 for 10%): ").strip()
)

input_class = input("Enter the class you want to detect: ").strip()

# Run the splitter
split_dataset_with_labels(input_dir, train_ratio, test_ratio, valid_ratio, input_class)

create_yaml_file(
    output_dir="output", class_names=input_class, yaml_file_path="dataset.yaml"
)

model = YOLO("yolo11n.pt")

model.train(
    data="dataset.yaml",  # Path to your .yaml file created earlier
    epochs=50,  # Number of epochs to train
)
