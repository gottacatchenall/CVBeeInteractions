import pandas as pd
import numpy as np
import torch
import uuid
import os
import time
import argparse

from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

parser = argparse.ArgumentParser(description='Cropping iNaturalist Images with Zero-Shot Object Detection')
parser.add_argument('--species', type=int, default=0, help='')
parser.add_argument('--cluster', action='store_true')


def get_species_names(base_path):
    sp_list = [f for f in os.listdir(os.path.join(base_path, "data", "img")) if not f.startswith('.')]
    sp_list.sort()
    return sp_list

def load_species_image_metadata(base_path, species_name):
    metadata = pd.read_csv(os.path.join(base_path, "data", "img",species_name, "_metadata.csv"))
    return metadata
   
def process_image(processor, model, image, prompt, device):  
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
        return result
    except:
        pass


def crop_image(image, bbox):
    x_min, y_min, x_max, y_max  = bbox
    xm, ym, xM, yM = [int(np.floor(i.cpu())) for i in [x_min, y_min, x_max, y_max]]
    cropped_img = image.crop((xm, ym, xM, yM))
    return cropped_img


def process_species_images(processor, model, base_path, species_name, device):
    metadata_df = load_species_image_metadata(base_path, species_name)
    image_paths = [os.path.join(base_path, image_path) for image_path in metadata_df.image]

    outdir_path = os.path.join(base_path, "data", "processed_img", species_name)
    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)

    prompt = "a bee." if "Bombus" in species_name else "a flowering plant."

    metadata = []
    total_time = 0
    for (i,img_path) in enumerate(image_paths):
        start = time.time()
        image = Image.open(img_path)
        result = process_image(processor, model, image, prompt, device)

        if result != None: 
            for box in result["boxes"]:
                cropped_img = crop_image(image, box)
                img_uuid = str(uuid.uuid4())
                img_path = os.path.join(outdir_path, img_uuid + '.jpg')
                cropped_img.save(img_path)
                obj = {
                    "path": img_path,
                    "user_id": metadata_df["user_id"][i],
                    "username": metadata_df["username"][i],
                    "observation_id": metadata_df["user_id"][i]
                }
                metadata.append(obj)
        
        end = time.time()
        total_time += end - start

    print("Avg time: %f" % total_time / len(image_paths))
    print("Total time: %f" % total_time)
    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(outdir_path, "_metadata.csv"), index=False)



def main():
    args = parser.parse_args()  
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else "./"

    model_id = "IDEA-Research/grounding-dino-base"
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)

    species_name = get_species_names(base_path)[args.species]
    process_species_images(processor, model, base_path, species_name, device)

if __name__=='__main__':
   main()

