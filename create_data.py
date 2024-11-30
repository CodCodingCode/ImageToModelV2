from transformers import AutoProcessor, AutoModelForCausalLM
from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
from PIL import Image
import requests
import copy
import torch


def initialize_groundingdino():
    model = load_model(
        "config.py",  # Path to the GroundingDINO config file
        "groundingdino_swint_ogc.pth",  # Path to the checkpoint file
    )
    return model


# Runs the detection using GroundingDINO
def run_example(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_groundingdino()

    # Load the image
    image_source, image = load_image(image_path)
    height, width = image_source.shape[0], image_source.shape[1]

    # Perform predictions
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=device,
    )

    # Return the results
    return boxes, logits, phrases, width, height


def convert_to_od_format(boxes, logits, labels):
    """
    Converts GroundingDINO outputs into object detection format.
    Args:
        boxes: Tensor of bounding boxes (normalized coordinates).
        logits: Tensor of confidence scores.
        labels: List of string labels.
    Returns:
        Dictionary with "bboxes", "scores", and "labels".
    """

    info = {
        "bboxes": boxes,
        "scores": logits,
        "labels": labels,
    }

    bboxes_list = info["bboxes"].tolist()
    od_results = {"bboxes": bboxes_list, "labels": labels}

    return od_results


def test():
    # Image and Prompt
    IMAGE_PATH = "images/514284-honda-civic-hyundai-tucson-named-ajac-s-2022-car-and-suv-of-the-year_jpg.rf.afdf95c1f4dda173c59ff872c30f826d.jpg"
    TEXT_PROMPT = "chair . person . dog . car ."

    # Run detection
    boxes, logits, phrases, width, length = run_example(
        image_path=IMAGE_PATH,
        text_prompt=TEXT_PROMPT,
        box_threshold=0.35,
        text_threshold=0.25,
    )
    print(width, length)

    # Convert to object detection format
    od_results = convert_to_od_format(boxes, logits, phrases)
    print(f"Object Detection Results: {od_results}")

    # Optional: Annotate and save the image
    image_source, _ = load_image(IMAGE_PATH)
    annotated_frame = annotate(
        image_source=image_source, boxes=boxes, logits=logits, phrases=phrases
    )
    cv2.imwrite("annotated_image.jpg", annotated_frame)


if __name__ == "__main__":
    test()
