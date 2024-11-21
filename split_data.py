import os
import shutil
from math import floor
from random import shuffle


def split_dataset(input_dir, output_dir, train_ratio, test_ratio, valid_ratio):
    """
    Splits images into train, test, and validation sets.

    Args:
        input_dir (str): Path to the input folder containing images.
        output_dir (str): Path to the output folder to save the splits.
        train_ratio (float): Proportion of images for training.
        test_ratio (float): Proportion of images for testing.
        valid_ratio (float): Proportion of images for validation.
    """
    # Validate ratios
    if not abs(train_ratio + test_ratio + valid_ratio - 1.0) < 1e-6:
        raise ValueError("The sum of train, test, and valid ratios must equal 1.")

    # List all image files
    images = [
        f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))
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

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    valid_dir = os.path.join(output_dir, "valid")

    for directory in [train_dir, test_dir, valid_dir]:
        os.makedirs(directory, exist_ok=True)

    # Copy images to respective directories
    def copy_files(file_list, target_dir):
        for file_name in file_list:
            shutil.copy(
                os.path.join(input_dir, file_name), os.path.join(target_dir, file_name)
            )

    copy_files(train_images, train_dir)
    copy_files(test_images, test_dir)
    copy_files(valid_images, valid_dir)

    print(f"Dataset split completed:")
    print(f"Train: {len(train_images)} images")
    print(f"Test: {len(test_images)} images")
    print(f"Valid: {len(valid_images)} images")


if __name__ == "__main__":
    # User inputs
    input_dir = input("Enter the path to the folder containing images: ").strip()
    output_dir = input("Enter the output folder path for the splits: ").strip()

    train_ratio = float(
        input("Enter the train split ratio (e.g., 0.7 for 70%): ").strip()
    )
    test_ratio = float(
        input("Enter the test split ratio (e.g., 0.2 for 20%): ").strip()
    )
    valid_ratio = float(
        input("Enter the validation split ratio (e.g., 0.1 for 10%): ").strip()
    )

    # Run the splitter
    split_dataset(input_dir, output_dir, train_ratio, test_ratio, valid_ratio)
