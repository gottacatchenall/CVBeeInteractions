import pandas as pd
import numpy as np
import torch
import uuid
import os
import time
import argparse


from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 

def get_species_names(base_path):
    sp_list = [f for f in os.listdir(base_path) if not f.startswith('.')]
    sp_list.sort()
    return sp_list

def load_species_image_metadata(base_path, img_dir, species_name):
    metadata = pd.read_csv(os.path.join(base_path, "data", img_dir, species_name, "_metadata.csv"))
    return metadata
   
def process_image(processor, model, image, prompt, device, filepath):  
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
            text_threshold = 0.3,
            target_sizes=[image.size[::-1]]
        )
        result = results[0]
        for box, score, labels in zip(result["boxes"], result["scores"], result["labels"]):
            box = [round(x, 2) for x in box.tolist()]
        return result
    except Exception as err:
        print(f"Unexpected {err=} for file {filepath=} ")



def crop_image(image, bbox):
    x_min, y_min, x_max, y_max  = bbox
    xm, ym, xM, yM = [int(np.floor(i.cpu())) for i in [x_min, y_min, x_max, y_max]]
    cropped_img = image.crop((xm, ym, xM, yM))
    return cropped_img


def process_species_images(processor, model, base_path, img_dir, species_name, device, batch_size=16, outdir_name = "cropped_more_bombus"):
    metadata_df = load_species_image_metadata(base_path, img_dir, species_name)
    
    image_paths = [os.path.join(base_path, image_path) for image_path in metadata_df.image]

    outdir_path = os.path.join(base_path, "data", outdir_name, species_name)
    if not os.path.exists(outdir_path):
        os.mkdir(os.path.join(base_path, "data", outdir_name))
    if not os.path.exists(outdir_path):
        os.mkdir(outdir_path)

    prompt = "a bee."
    metadata = []

    for i,img_path in enumerate(image_paths):
        # Load images       
        img = Image.open(img_path).convert("RGB")

        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            text_threshold=0.3,
            target_sizes=[img.size[::-1]]
        )

        # Handle detections + async saving
        for result in results:
            for box in result["boxes"]:
                cropped_img = crop_image(img, box)
                img_uuid = str(uuid.uuid4())
                save_path = os.path.join(outdir_path, img_uuid + ".jpg")
                cropped_img.save(save_path)
                obj = {
                    "path": save_path,
                    "user_id": metadata_df["user_id"].iloc[i],
                    "username": metadata_df["username"].iloc[i],
                    "observation_id": metadata_df["user_id"].iloc[i],
                }
                metadata.append(obj)

    df = pd.DataFrame(metadata)
    df.to_csv(os.path.join(outdir_path, "_metadata.csv"), index=False)

def main(args):
    
    base_path = os.path.join("/scratch", "mcatchen", "iNatImages") if args.cluster else os.path.join("./")
 
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    model_id = "IDEA-Research/grounding-dino-base"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id, local_files_only=True).to(device)
    #model = torch.compile(model)


    species_names = get_species_names(os.path.join(base_path, "data", args.img_dir))
    for species_name in species_names:
        process_species_images(processor, model, base_path, args.img_dir, species_name, device, batch_size=args.batch_size)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Cropping iNaturalist Images with Zero-Shot Object Detection')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--img_dir', default='more_bees')
    parser.add_argument('--cluster', action='store_true')
    parser.add_argument('--batch-size', type=int, default=16, help="Batch size for inference")

    args = parser.parse_args()  
    main(args)

