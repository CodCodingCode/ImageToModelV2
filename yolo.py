from ultralytics import YOLO

model = YOLO("best.pt")


model.predict(
    source="/Users/owner/Downloads/coding projects/ImageToModelV2/output/test/images/2023-bmw-760i-xdrive-101-1650340309_jpg.rf.18ccac554e3c0af079afc72a95cd7540.jpg",
    save=True,
    conf=0.2,
    show=True,
)
