import argparse
from yelp_scrape import scrape
from utils import create_scrape_list, coords_to_points, tag_resturant_json
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument("--scrape", action='store_true', required=False)
parser.add_argument("--num_reviews", type=int, required=False)
parser.add_argument("--zone", type=int, required=False)
args = parser.parse_args()

if __name__ == "__main__":

    if args.scrape:
        with open("data/inspected_restuarants.json", "r") as rawfile:
            data = json.load(rawfile)

        if args.num_reviews: 
            num_reviews = args.num_reviews
        else:
            num_reviews = 10
    
        tagged_data = tag_resturant_json(data)
        coords_dict = coords_to_points('data/points.json')
        out_folder = "data/scraped"
        os.makedirs(out_folder, exist_ok=True)

        if Path('data/checklist.json').is_file():
            with open('data/checklist.json', 'r') as f:
                checklist = json.load(f)
        else:
            checklist = create_scrape_list(tagged_data)
        
        if args.zone:
            # specifiy coordinate set to scrape
            scrape(data, out_folder, args.zone, coords_dict, num_reviews, checklist)

        else:
            # scrape everything
            for zone in coords_dict.keys():
                scrape(data, out_folder, zone, coords_dict, num_reviews, checklist)


