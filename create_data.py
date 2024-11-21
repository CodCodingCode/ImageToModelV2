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
def run_example(task_prompt, image="", text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(torch.float16)
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )

    return parsed_answer


def convert_to_od_format(data):
    """
    Converts a dictionary with 'bboxes' and 'bboxes_labels' into a dictionary with separate 'bboxes' and 'labels' keys.

    Parameters:
    - data: The input dictionary with 'bboxes', 'bboxes_labels', 'polygons', and 'polygons_labels' keys.

    Returns:
    - A dictionary with 'bboxes' and 'labels' keys formatted for object detection results.
    """
    # Extract bounding boxes and labels
    bboxes = data.get("bboxes", [])
    labels = data.get("bboxes_labels", [])

    # Construct the output format
    od_results = {"bboxes": bboxes, "labels": labels}

    return od_results


def test():
    task_prompt = "<OPEN_VOCABULARY_DETECTION>"
    text_input = "car"
    url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    image = Image.open(requests.get(url, stream=True).raw)
    results = run_example(task_prompt, text_input=text_input, image=image)
    print(results)  # print result
    data = convert_to_od_format(results["<OPEN_VOCABULARY_DETECTION>"])
    print(data)  # print data


if __name__ == "__main__":
    test()
