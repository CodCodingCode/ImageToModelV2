import os
import shutil
import yaml
from math import floor
from random import shuffle
from convert_to_yolo import convert_to_yolo_format
from create_data import run_example, convert_to_od_format
from ultralytics import YOLO
import cv2
from groundingdino.util.inference import load_model, load_image, predict, annotate
import streamlit as st  # Import Streamlit


def generate_label_for_image(image_path, text_input, label):

    # Convert to object detection format
    boxes, logits, phrases, width, length = run_example(
        image_path, text_prompt=text_input, box_threshold=0.35, text_threshold=0.25
    )

    data = convert_to_od_format(boxes, logits, phrases)
    print(data, width, length)

    message = convert_to_yolo_format(
        data, img_width=width, img_height=length, label=label
    )
    print(message)

    return message


def split_dataset_with_labels(
    input_dir, train_ratio, test_ratio, valid_ratio, input_class, progress_bar
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
        for i, file_name in enumerate(file_list):
            image_src_path = os.path.join(input_dir, file_name)

            # Check if the file is a valid image
            if not os.path.isfile(image_src_path):
                print(f"File not found: {image_src_path}")
                continue

            try:
                label_data = generate_label_for_image(
                    image_src_path,
                    input_class,
                    input_class,
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

            # Update progress bar
            progress_bar.progress((i + 1) / len(file_list))

    # Process each split
    process_files(train_images, "train")
    process_files(test_images, "test")
    process_files(valid_images, "valid")

    # Finalize progress bar
    progress_bar.progress(1.0)

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

# Streamlit UI for user inputs
st.title("Dataset Preparation for YOLO")
uploaded_files = st.file_uploader(
    "Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary location or process it directly
        with open(os.path.join("temp_dir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())

input_dir = st.text_input("Enter the path to the folder containing images:")
train_ratio = st.number_input(
    "Enter the train split ratio (e.g., 0.7 for 70%):",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
)
valid_ratio = st.number_input(
    "Enter the validation split ratio (e.g., 0.1 for 10%):",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
)
test_ratio = st.number_input(
    "Enter the test split ratio (e.g., 0.2 for 20%):",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
)
input_class = st.text_input("Enter the class you want to detect:")

# Button to run the splitter
if st.button("Split Dataset"):
    if input_dir and input_class:
        print(f"Input directory: {input_dir}")  # Debugging line
        if not os.path.exists(input_dir):
            st.error(
                f"Directory does not exist: {input_dir}"
            )  # Error message for the user
        else:
            progress_bar = st.progress(0)  # Initialize progress bar
            split_dataset_with_labels(
                input_dir,
                train_ratio,
                test_ratio,
                valid_ratio,
                input_class,
                progress_bar,
            )
            create_yaml_file(
                output_dir="output",
                class_names=[input_class],
                yaml_file_path=yaml_file_path,
            )
            st.success("Dataset split completed and YAML file created.")
    else:
        st.error("Please fill in all fields.")

model_path = ""
# Button to train the model
if st.button("Train Model"):
    model = YOLO("yolo11n.pt")
    model_path = "/Users/owner/Downloads/coding projects/ultralytics/Drone_proj/runs/detect/train57/weights/best.pt"
    progress_bar = st.progress(0)  # Initialize progress bar
    model.train(data=yaml_file_path, epochs=30)  # Train for one epoch at a time
    st.success("Model training completed.")
    st.write(f"Trained model saved at: {model_path}")  # Display the model path

if model_path and os.path.exists(model_path):
    with open(model_path, "rb") as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name=os.path.basename(model_path),
            mime="application/octet-stream",
        )
