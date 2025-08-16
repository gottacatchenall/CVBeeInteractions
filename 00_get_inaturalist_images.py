from pyinaturalist import *
from dotenv import load_dotenv

import uuid
import requests
import os 
import pandas as pd
import numpy as np
import logging

import logging

logging.getLogger("pyinaturalist").setLevel(logging.CRITICAL)

load_dotenv()  

INAT_API_KEY = os.getenv("INATURALIST_API_KEY")

def load_species_list():
    dir_path = os.path.join("data", "species_lists")
    dfs = [pd.read_csv(os.path.join(dir_path,x)) for x in os.listdir(dir_path)]
    return pd.concat(dfs)["species"].to_list()

def load_taxon_ids():
    df = pd.DataFrame(columns=['species_name', 'species_id'])
    ct = 0
    for sp_name in species_list:
        response = get_taxa(sp_name, rank='species')
        taxa = Taxon.from_json_list(response)[0]
        df.loc[ct] = [sp_name, taxa.id]
        ct += 1
    return df

def save_images(observations, dirpath):
    img_metadata = []
    for (i,o) in enumerate(observations):
        img_url = o.photos[0].url.replace("square", "large")
        img_uuid = str(uuid.uuid4())
        img_path = os.path.join(dirpath, img_uuid + '.jpg')

        headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.106 Safari/537.36'} 

        img_data = requests.get(img_url, headers=headers).content

        with open(img_path, 'wb') as handler:
            handler.write(img_data)
        
        this_img = {
            'species': o.taxon.full_name,
            'image': img_path,
            'observation_id': o.id,
            'licence': o.license_code,
            'taxa_id': o.taxon.id,
            'username': o.user.login,
            'user_id': o.user.id,
            'longitude': '' if o.location == None else o.location[1],
            'latitude': '' if o.location == None else o.location[0]
        }
        img_metadata.append(this_img)

    return img_metadata


def download_single_species_images(species_name, max_obs=1000):
    dirpath = os.path.join("data", "large_img", species_name)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)

    per_page = 200
    max_page = int(np.floor(max_obs / 200))

    all_metadata = []

    for page in range(1,max_page+1):
        print("Species: %s \t | Page: %i" % (species_name, page))
        response = get_observations(
            taxon_name=species_name, 
            photos=True, 
            photo_license=['CC0', 'CC-BY', 'CC-BY-NC'], 
            page=page,
            per_page=per_page,
            access_token=INAT_API_KEY
        )
        obs = Observation.from_json_list(response)
        if len(obs) == 0:
            break
        metadata = save_images(obs, dirpath)
        all_metadata += metadata

    df = pd.DataFrame(all_metadata)
    df.to_csv(os.path.join(dirpath, "_metadata.csv"), index=False)
    
       

def download_species_images(species_names):
    for species_name in species_names:
        download_single_species_images(species_name)
       

species_list = load_species_list()
download_species_images(species_list)
