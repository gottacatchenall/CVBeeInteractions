import polars as pl
import os
import sys 
import datetime
from datetime import datetime, timedelta
import numpy as np
import xgboost as xgb
import webdataset as wds
import glob 
import torch
import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import scipy

def load_raw_interaction_dataframes():
    raw_int_data_path = os.path.join("data", "raw_interactions")

    def get_dataset_name_from_path(x):
        return x.split(".")[0]
    def read_dataframe(filename):
        return pl.read_csv(os.path.join(raw_int_data_path, filename))
    
    raw_dfs = {
        get_dataset_name_from_path(x): read_dataframe(x)
        for x in os.listdir(raw_int_data_path)
    }
    return raw_dfs

def clean_pikespeak(pikes_df):
    return pl.DataFrame({
        "year": pikes_df["year"], 
        "month": pikes_df["month.x"], 
        "day": pikes_df["day.x"], 
        "plant": pikes_df["ack.nam"],  
        "bee": pikes_df["pol_sp"],
        "site": ["gothic" for x in pikes_df.rows()]
    })

def clean_gothic(gothic_df):
    def convert_to_datetime(year, doy):
        return datetime(year, 1, 1) + timedelta(days=doy - 1)

    return pl.DataFrame({
        "year": gothic_df["year"], 
        "month": [convert_to_datetime(x[0], x[2]).month for x in gothic_df.iter_rows()], 
        "day": [convert_to_datetime(x[0], x[2]).day for x in gothic_df.iter_rows()], 
        "plant": [x.split(".")[0]+" "+x.split(".")[1] for x in gothic_df["plant.species"]],
        "bee": ["Bombus " + x for x in gothic_df["species"]],
        "site": ["gothic" for x in gothic_df.rows()]
    })

def clean_elk(elk_df):
    return pl.DataFrame({
        "year": elk_df["Year"], 
        "month": [datetime.strptime(x, "%m/%d/%Y").month for x in elk_df["Date"]], 
        "day": [datetime.strptime(x, "%m/%d/%Y").day for x in elk_df["Date"]], 
        "plant": elk_df["Plant species name"],
        "bee": elk_df["Insect species name"],
        "site": ["elkmeadows" for x in elk_df.rows()]
    })
 
def clean_dfs(raw_dfs):
    return {
        "elkmeadows": clean_elk(raw_dfs["elkmeadows"]),
        "pikespeak": clean_pikespeak(raw_dfs["pikespeak"]),
        "gothic": clean_gothic(raw_dfs["gothic"])
    }

def combine_clean_dfs():
    return clean_dfs(load_raw_interaction_dataframes())

def get_dataframe():
    dfs = combine_clean_dfs()
    total_df = pl.concat(dfs.values()).drop_nulls()
    total_df = total_df.with_columns([
    pl.struct(total_df.columns).map_elements(lambda x: x["plant"].split(" ")[0] +" "+x["plant"].split(" ")[1], return_dtype=str ).alias("plant")
    ])
    total_df= total_df.filter(pl.col("bee") != "Bombus (fervidus) californicus")
    
    plant_species = os.listdir(os.path.join("data", "plant_img"))

    total_df = total_df.remove(
        pl.col("plant").is_in(plant_species) == False
    )

    return total_df 


def get_metaweb():
    df = get_dataframe()
    plant_species, bee_species = df["plant"].unique().sort(), df["bee"].unique().sort()
    metaweb = np.zeros((len(plant_species), len(bee_species)))

    bee2idx = {b:i for i,b in enumerate(bee_species)}
    plant2idx = {b:i for i,b in enumerate(plant_species)}

    idx2bee = {v:k for k,v in bee2idx.items()}
    idx2plant = {v:k for k,v in plant2idx.items()}

    for r in df.iter_rows():
        _, _, _, plant, bee, _ = r
        metaweb[plant2idx[plant],bee2idx[bee]] = 1
    
    interaction_dicts = []

    ct = 0
    for plant_idx, plant_row in enumerate(metaweb):
        for bee_idx, interaction_bit in enumerate(plant_row):
            interaction_dicts.append({
                "plant": idx2plant[plant_idx],
                "bee": idx2bee[bee_idx],
                "interaction": int(interaction_bit)
            })
            ct += 1

    

    return metaweb, pl.DataFrame(interaction_dicts), plant2idx, bee2idx

mw, df, plant2idx, bee2idx = get_metaweb()
df.write_csv(os.path.join("data", "interactions.csv"))

with open(os.path.join("data", "plant_labels.json"), "w") as json_file:
    json.dump(plant2idx, json_file, indent=4) 

with open(os.path.join("data", "bee_labels.json"), "w") as json_file:
    json.dump(bee2idx, json_file, indent=4) 

"""

def get_class_dict(image_dir, num_classes):
    def preprocess(data):
        img_bytes, meta = data
        class_label = meta["label"]
        class_name = meta["class_name"]
        return class_label, class_name

    train_dataset = (
        wds.WebDataset(glob.glob(f"{image_dir}/train*.tar"))
            .decode()
            .to_tuple("jpg", "json")
    ).map(preprocess)

    label_to_name = {}
    
    it = train_dataset.iterator()
    while len(label_to_name) < num_classes:
        l, n = next(it)
        label_to_name[l] = n
    return label_to_name

def get_bee_labels():
    return {v: k for k, v in  get_class_dict("./data/bombus_wds", 19).items()}

def get_plant_labels():
    return {v: k for k, v in  get_class_dict("./data/plant_wds", 158).items()}

def build_labels():
    bl, pl = get_bee_labels(), get_plant_labels()
    return bl, pl




def get_metaweb():
    bee_labels, plant_labels = build_labels()
    df = get_dataframe()
    metaweb = np.zeros((len(plant_labels), len(bee_labels)))

    for r in df.iter_rows():
        _, _, _, plant, bee, _ = r
        pi, bi = plant_labels[plant], bee_labels[bee]    
        metaweb[pi,bi] = 1
    return metaweb

def write_metaweb(outpath):
    bee_labels, plant_labels = build_labels()
    df = get_dataframe()
    metaweb = get_metaweb()
    
    bee_idx_to_name = {v: k for k, v in  bee_labels.items()}
    plant_idx_to_name = {v: k for k, v in  plant_labels.items()}

    dicts = []
    for plant_idx in range(metaweb.shape[0]):
        for bee_idx in range(metaweb.shape[1]):
            dicts.append({ 
                "pollinator_index": bee_idx,
                "plant_index": plant_idx,
                "pollinator_name": bee_idx_to_name[bee_idx],
                "plant_name": plant_idx_to_name[plant_idx],
                "interaction": int(metaweb[plant_idx, bee_idx]),
            })

    pl.DataFrame(dicts).write_csv(outpath)

#write_metaweb("./data/interactions.csv")


def build_feat_and_labs(metaweb, bee_embed, plant_embed):
    bee_labels, plant_labels = build_labels()

    embed_dim = 128
    X = np.zeros((np.prod(metaweb.shape), 2*embed_dim))
    y = np.zeros(np.prod(metaweb.shape))
    idxs = np.zeros((np.prod(metaweb.shape), 2))

    cursor = 0
    for (bee, bi) in bee_labels.items():
        for (plant, pi) in plant_labels.items():
            X[cursor,:embed_dim] = bee_embed[bee]
            X[cursor,embed_dim:2*embed_dim] = plant_embed[plant]
        
            y[cursor] = bool(metaweb[pi,bi])
            idxs[cursor, :] = [bi, pi]
            cursor += 1

    return idxs,X,y

import torch
import sklearn
import xgboost as xgb
from sklearn.datasets import  make_classification
from sklearn.metrics import roc_auc_score, average_precision_score


def get_pseudoembeds(embed_dim):
    bee_labels, plant_labels = build_labels()
    bee_embed = {}
    for bee in bee_labels.keys():
        bee_embed[bee] = np.random.standard_normal(embed_dim)
    plant_embed = {}
    for plant in plant_labels.keys():
        plant_embed[plant] = np.random.standard_normal(embed_dim)

    return bee_embed, plant_embed

def run_xgb(X,y, reps=64):
    # Show all messages, including ones pertaining to debugging
    xgb.set_config(verbosity=0)
    reps =64
    prs = []
    rocs = []
    for n in range(reps):
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            X, y, test_size=0.2
        )


        num_pos = sum(y_train) 
        num_neg = (len(y_train) - sum(y_train))
        model = xgb.XGBClassifier(
            eval_metric="auc",  # avoid warnings
            scale_pos_weight = num_neg / num_pos,
        )
        # Fit the model, test sets are used for early stopping.
        _ = model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 6. Compute ROC AUC and PR AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        rocs.append(roc_auc)

        p, r, t = sklearn.metrics.precision_recall_curve(y_test, y_proba)
        pr_auc = sklearn.metrics.auc(r,p)
        #pr_auc = average_precision_score(y_test, y_proba)
        prs.append(pr_auc)
    return prs, rocs
    #return np.mean(prs), np.mean(rocs)



bee_embed = torch.load("data/bees_embed.pt", map_location=torch.device('mps') )
plant_embed = torch.load("data/plants_embed.pt" , map_location=torch.device('mps') )


def first_zero_sum(tensor, dim=0):
    col_sums = tensor.sum(dim=dim)
    zero_indices = (col_sums == 0).nonzero(as_tuple=True)[0]
    return zero_indices[0].item() if len(zero_indices) > 0 else None

def get_mean_emb(embed_dict):
    mean_emb = {}
    for k,v in embed_dict.items():
        first_zero_col = first_zero_sum(v, dim=1)
        e = v[:first_zero_col,:]
        e_bar = torch.mean(e, 0).cpu().detach()
        mean_emb[k] = e_bar
    return mean_emb



mean_plants = get_mean_emb(plant_embed)
mean_bees = get_mean_emb(bee_embed)


metaweb = get_metaweb()

idx, Xreal, yreal = build_feat_and_labs(metaweb, mean_bees, mean_plants)



bee_labels, plant_labels = build_labels()

bee_id2sp = {int(v):k for k,v in bee_labels.items()}
plant_id2sp = {int(v):k for k,v in plant_labels.items()}

with open("data/bee_labels.json", "w") as f:
    json.dump(bee_id2sp, f, indent=4)
with open("data/plant_labels.json", "w") as f:
    json.dump(plant_id2sp, f, indent=4)




with open('data/plant_labels.json') as json_file:
    data = json.load(json_file)




def zero_shot_split(species_idx, X, y, bee_holdouts=3, plant_holdouts=10):
    bee_idx = np.unique(species_idx[:,0])
    plant_idx = np.unique(species_idx[:,1])

    np.random.shuffle(bee_idx)   
    np.random.shuffle(plant_idx)   

    heldout_bees = bee_idx[:bee_holdouts]
    heldout_plants = plant_idx[:plant_holdouts]

    bee_train_mask = [x not in heldout_bees for x in idx[:,0]]
    plant_train_mask = [x not in heldout_plants for x in idx[:,1]]
   
    train_mask = [x and y for x, y in zip(bee_train_mask, plant_train_mask)]
    test_mask = [not x for x in train_mask]

    return Xreal[train_mask], yreal[train_mask], Xreal[test_mask], yreal[test_mask]


def batch_zero_shot(
    X, y,
    reps = 64,
    bee_holdouts=0, 
    plant_holdouts=10
):

    prs = []
    rocs = []
    for n in range(reps):
        X_train, y_train, X_test, y_test = zero_shot_split(idx, X, y, bee_holdouts=bee_holdouts, plant_holdouts=plant_holdouts)

        model = xgb.XGBClassifier(
            eval_metric="logloss",  # avoid warnings
        )
        # Fit the model, test sets are used for early stopping.
        _ = model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_proba = model.predict_proba(X_test)[:, 1]

        # 6. Compute ROC AUC and PR AUC
        roc_auc = roc_auc_score(y_test, y_proba)
        rocs.append(roc_auc)
        pr_auc = average_precision_score(y_test, y_proba)
        prs.append(pr_auc)
    return np.mean(prs), np.mean(rocs)




fake_bee_emb, fake_plant_emb = get_pseudoembeds(128)
idx, Xfake, yfake = build_feat_and_labs(metaweb, fake_bee_emb, fake_plant_emb)

realpr, realroc = run_xgb(Xreal,yreal, reps=256)
fakepr, fakeroc = run_xgb(Xfake,yfake, reps=256)

print(f"Real PRAUC: {np.mean(realpr)}, ROCAUC {np.mean(realroc)}")
print(f"Fake PRAUC: {np.mean(fakepr)}, ROCAUC {np.mean(fakeroc)}")


scipy.stats.ttest_ind(realpr, fakepr)
scipy.stats.ttest_ind(realroc, fakeroc)

"""