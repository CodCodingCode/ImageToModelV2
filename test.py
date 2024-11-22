from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="dataset.yaml",  # Path to your .yaml file created earlier
    epochs=50,  # Number of epochs to train
)
