import polars as pl
import os
import sys 
import datetime
from datetime import datetime, timedelta


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
    

    return total_df 

    
df = get_dataframe()
