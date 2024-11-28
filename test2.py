from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = load_model(
    "config.py",
    "groundingdino_swint_ogc.pth",
)
IMAGE_PATH = "images/514284-honda-civic-hyundai-tucson-named-ajac-s-2022-car-and-suv-of-the-year_jpg.rf.afdf95c1f4dda173c59ff872c30f826d.jpg"
TEXT_PROMPT = "chair . person . dog . car ."
BOX_TRESHOLD = 0.35
TEXT_TRESHOLD = 0.25

image_source, image = load_image(IMAGE_PATH)

boxes, logits, phrases = predict(
    model=model,
    image=image,
    caption=TEXT_PROMPT,
    box_threshold=BOX_TRESHOLD,
    text_threshold=TEXT_TRESHOLD,
    device=device,
)


print(boxes, logits, phrases)
annotated_frame = annotate(
    image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
)
cv2.imwrite("annotated_image.jpg", annotated_frame)
