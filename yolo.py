from ultralytics import YOLO

model = YOLO("yolo11n.pt")


model.predict(
    source="output/valid/images/514284-honda-civic-hyundai-tucson-named-ajac-s-2022-car-and-suv-of-the-year_jpg.rf.afdf95c1f4dda173c59ff872c30f826d.jpg",
    save=True,
    conf=0.2,
    show=True,
)