import torch
from segment_anything import SamPredictor, sam_model_registry
from groundingdino.util.inference import load_model, predict
import cv2
import matplotlib.pyplot as plt


# Set the device dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the pre-trained SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Download from SAM's GitHub repo
sam_model = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam_model.to(device)  # Move SAM model to the correct device
sam_predictor = SamPredictor(sam_model)


# Load the pre-trained Grounding DINO model
dino_checkpoint = "groundingdino_swint_ogc.pth"  # Download from Grounding DINO's repo
dino_config = "config.py"
dino_model = load_model(dino_config, dino_checkpoint)
dino_model.to(device)  # Move Grounding DINO model to the correct device


# Load and preprocess the image
image_path = "images/514284-honda-civic-hyundai-tucson-named-ajac-s-2022-car-and-suv-of-the-year_jpg.rf.afdf95c1f4dda173c59ff872c30f826d.jpg"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Convert the image to a PyTorch tensor
image_tensor = (
    torch.from_numpy(image_rgb).permute(2, 0, 1).float()
)  # Change shape from HWC to CHW


# Predict ROIs with SAM
sam_predictor.set_image(image_tensor)
masks, _, _ = sam_predictor.predict(
    point_coords=None,  # No specific points; predict all regions
    point_labels=None,
    multimask_output=True,
)


print(masks)
# Define the prompt
prompt = "car . person ."


# Use DINO for label prediction
boxes, labels, _ = predict(
    model=dino_model,
    image=image_tensor.to(device),  # Move the tensor to the correct device
    caption=prompt,  # Pass the prompt directly as a string
    box_threshold=0.3,  # Confidence threshold
    text_threshold=0.25,
    device=device,
)


# Define utility function to convert bounding boxes
def box_cxcywh_to_xyxy(boxes):
    """
    Convert bounding boxes from center-x, center-y, width, height (cxcywh) format
    to x1, y1, x2, y2 format.
    """
    x_c, y_c, w, h = boxes.unbind(-1)
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return torch.stack((x1, y1, x2, y2), dim=-1)


print(boxes, labels)
# Convert the boxes to a format compatible with SAM
boxes = box_cxcywh_to_xyxy(boxes)


# Draw bounding boxes and labels
for box, label in zip(boxes, labels):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
    )


# Show the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
