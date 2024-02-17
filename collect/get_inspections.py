from zipfile import ZipFile
from pathlib import Path
import pandas as pd
import polars as pl
import argparse, json



def read_inspections():
    # ONLY NEED TO DO THIS ONCE
    # download all data from this website: https://data.cityofchicago.org/Health-Human-Services/Food-Inspections/4ijn-s7e5/about_data
    # and save locally to data/ directory as inspections.csv, and compress as zip file
    zip_file = ZipFile('data/inspections.csv.zip')
    df = pd.read_csv(zip_file.open('inspections.csv'))
    df.to_parquet("data/inspections.parquet")
    return

def clean_inspections(inspections_fp):
    # POLARS VERSION
    # df = pl.read_parquet(inspections_fp)
    
    # #filtering out restaurants that do not have a license number
    # df = df.drop_nulls("License #").filter(pl.col("License #") > 0)

    # #convert inspection date to datetime object and add new col with the previous inspection date
    # df = df.with_columns([pl.col('Inspection Date').str.to_datetime("%m/%d/%y")]).sort(pl.col('Inspection Date')).sort(pl.col('License #'))
    # df = df.with_columns([pl.col('Inspection Date').shift().over('License #', "DBA Name").alias('Prev Inspection Date')])

    # Read the parquet file into a pandas DataFrame
    df = pd.read_parquet(inspections_fp)

    # Filtering out restaurants that do not have a license number or it is 0
    df = df.dropna(subset="License #")
    df = df.loc[df.loc[:,"License #"] > 0,:]

    # Convert inspection date to datetime object and add new column with the previous inspection date
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%y")
    df = df.sort_values(by="Inspection Date").sort_values(by="License #")
    df["Prev Inspection Date"] = df.groupby(["License #", "DBA Name"])["Inspection Date"].shift()

    return df


def clean_restaurants(restaurants_fp):
    
    #read json file
    with open(restaurants_fp, "r") as infile:
        data = json.load(infile)
    
    # standardize json into df format and concat into one df
    dfs = []
    for k,v in data.items():
        df = pd.json_normalize(v, max_level=4)
        df['point'] = k
        dfs.append(df)

    restaurants = pd.concat(dfs)

    # filter out restaurants that have no reviews or are closed
    restaurants = restaurants.loc[restaurants.loc[:,"is_closed"] == False,:]
    restaurants = restaurants.loc[restaurants.loc[:,"review_count"] > 0,:]

    # data pre-processing to increase match likelihood
    restaurants[['name', 
                 'location.address1', 
                 'location.address2', 
                 'location.address3', 
                 'location.city', 
                 'location.state']] = restaurants[
                     ['name', 
                        'location.address1', 
                        'location.address2', 
                        'location.address3', 
                        'location.city', 
                        'location.state']].map(str.upper, na_action='ignore')

    restaurants[['location.zip_code']] = restaurants[['location.zip_code']].astype('float64')
    
    return restaurants

def merge(inspections, restaurants):

    merged = pd.merge(left=inspections,
             right=restaurants, 
             how="inner",
             left_on=['Address'], #, 'City', 'State', 'Zip'], 
             right_on=['location.address1'] #, 
                        # 'location.city', 
                        # 'location.state', 
                        # 'location.zip_code']
    )
    return merged


def main(inspections_fp: str, restaurants_fp: str):
    inspections = clean_inspections(inspections_fp)
    restaurants = clean_restaurants(restaurants_fp)
    # print(inspections.loc[:,"Zip"])
    # print(restaurants.loc[:,"location.zip_code"])

    merged = merge(inspections, restaurants)
    print(merged)

    return 


def setup(args=None):    
    parser = argparse.ArgumentParser(description='Cleans inspections data.')
    parser.add_argument('--inspections_fp', required=True, type=Path, dest="inspections_fp", help="Path to inspections parquet file.")
    parser.add_argument('--restaurants_fp', required=True, type=Path, dest="restaurants_fp", help="Path to restaurants json file.")
    return parser.parse_args(args)


if __name__ == "__main__":
    main(**vars(setup()))
