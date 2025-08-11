import pandas as pd
import numpy as np
import torch
import uuid
import os
from PIL import Image

from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 


def load_species_image_metadata(species_name):
    metadata = pd.read_csv(os.path.join("data", "img", species_name, "_metadata.csv"))
    return metadata
   
def process_image(image, prompt):  
    try:  
        inputs = processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold = 0.4,
            text_threshold = 0.3,
            target_sizes=[image.size[::-1]]
        )
        result = results[0]
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
            print(f"Detected {labels} with confidence {round(score.item(), 3)} at location {box}")
        return result
    except:
        pass

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_results(img, result):
  fig, ax = plt.subplots(1)
  ax.imshow(img)

  detections = [
      {'box': result["boxes"][r].cpu(), 'label': result["labels"][r], 'score': result["scores"][r]}
      for r in range(len(result["scores"]))
  ]

  for det in detections:
    x_min, y_min, x_max, y_max = det['box']
    width = x_max - x_min
    height = y_max - y_min

    # Create a Rectangle patch
    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Add label and confidence score
    ax.text(x_min, y_min - 1, f"{det['label']} ({det['score']:.2f})", color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.7))

  fig.show()
"""


def crop_image(image, bbox):
    x_min, y_min, x_max, y_max  = bbox
    xm, ym, xM, yM = [int(np.floor(i.cpu())) for i in [x_min, y_min, x_max, y_max]]
    cropped_img = image.crop((xm, ym, xM, yM))
    return cropped_img


def process_species_images(species_name):
    dirpath = os.path.join("data", "processed_img", species_name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    metadf = load_species_image_metadata(species_name)
    prompt = "a bee." if "Bombus" in species_name else "a flowering plant."

    metadata = []

    for i,r in metadf.iterrows():
        img_path = r["image"]
        image = Image.open(img_path)
        result = process_image(image, prompt)

        if result != None: 
            for box in result["boxes"]:
                cropped_img = crop_image(image, box)
                img_uuid = str(uuid.uuid4())
                img_path = os.path.join(dirpath, img_uuid + '.jpg')
                cropped_img.save(img_path)

                obj = {
                    "path": img_path,
                    "user_id": r["user_id"],
                    "username": r["username"],
                    "observation_id": r["user_id"]
                }
                metadata.append(obj)
        
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(dirpath, "_metadata.csv"), index=False)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_id = "IDEA-Research/grounding-dino-tiny"
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

os.mkdir(os.path.join("data", "processed_img"))

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)


bombus = [x for x in os.listdir(os.path.join("data", "img")) if "Bombus" in x]

process_species_images(bombus[0])