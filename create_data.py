from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import requests
import copy
import torch


model_id = "microsoft/Florence-2-large"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype="auto"
).eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


# Runs the code in order to detect specific objects
def run_example(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25):
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = initialize_groundingdino()

    # Load the image
    image_source, image = load_image(image_path)

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
    return boxes, logits, phrases


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
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input = "car"
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    results = run_example(task_prompt=task_prompt, text_input=text_input, image=url)
    print(results)  # print result
    data = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])
    print(data)  # print data


if __name__ == "__main__":
    test()


"""
example output:
({'<OPEN_VOCABULARY_DETECTION>': {'bboxes': [[34.23999786376953, 160.0800018310547, 597.4400024414062, 371.7599792480469]], 'bboxes_labels': ['car'], 'polygons': [], 'polygons_labels': []}}, 640, 480)
"""
