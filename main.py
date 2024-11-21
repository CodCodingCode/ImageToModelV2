from split_data import split_dataset


input_dir = input("Enter the path to the folder containing images: ").strip()
output_dir = input("Enter the output folder path for the splits: ").strip()

train_ratio = float(input("Enter the train split ratio (e.g., 0.7 for 70%): ").strip())
test_ratio = float(input("Enter the test split ratio (e.g., 0.2 for 20%): ").strip())
valid_ratio = float(
    input("Enter the validation split ratio (e.g., 0.1 for 10%): ").strip()
)





# Run the splitter
split_dataset(input_dir, output_dir, train_ratio, test_ratio, valid_ratio)
test_path = f"{output_dir}/test"
valid_path = f"{output_dir}/valid"
train_path = f"{output_dir}/train"



print(test_path, train_path, valid_path)
