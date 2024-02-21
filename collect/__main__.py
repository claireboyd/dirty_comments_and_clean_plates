import argparse
from yelp_scrape import scrape
import json
import os

parser = argparse.ArgumentParser()
parser.add_argument("--scrape", action='store_true', required=False)
parser.add_argument("--num_reviews", type=int, required=False)
args = parser.parse_args()

if __name__ == "__main__":

    if args.scrape:
        with open("data/restuarant_pull.json", "r") as rawfile:
            data = json.load(rawfile)

            if args.num_reviews: 
                num_reviews = args.num_reviews
            else:
                num_reviews = 25

            out_folder = "data/scraper"
            os.makedirs(out_folder, exist_ok=True)
            all_coords = data.keys()
            for coords in all_coords:
                scrape(data, out_folder, coords, num_reviews, start=None, stop=None)
