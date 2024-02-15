from pathlib import Path
import argparse
import os, sys
import pprint
import requests
from dotenv import load_dotenv

#import vars from .env to os.environ
load_dotenv()
YELP_API_KEY = os.getenv('YELP_API_KEY')

# example args
lat = 41.791320
long = -87.592710
limit = 10

url = f"https://api.yelp.com/v3/businesses/search?latitude={lat}&longitude={long}&sort_by=best_match&limit={limit}"


def fetch_results(params, name):
    """
    params (dict) includes valid params for a search_query
    https://docs.developer.yelp.com/reference/v3_business_search

    name to save the run as: 
    """

    try:
        #build query
        base_url="https://api.yelp.com/v3/businesses/search?"

        #provide necessary headers
        headers = {"accept": "application/json",
                   "Authorization": f"Bearer {YELP_API_KEY}"}
        
        #get response and save
        response = requests.get(url=base_url, params=params, headers=headers)


        with open(f"data/restaurants_{name}.json", "w") as outfile:
            outfile.write(response.json())
        

    except Exception as e:
        return str(e)

def main():

    params = {
        "latitude": 41.791320,
        "longitude": -87.592710,
        "limit": 10
    }

    response = fetch_results(params, "test")

    print(response)


    
def setup(args=None):    
    parser = argparse.ArgumentParser(description='Gets JSON of Yelp Businesses from Lat/Long.')

    # ADD CLI ARGUMENTS HERE
    #parser.add_argument('--latlong_file', required=True, type=Path, dest="latlong_file", help="Path to file with lat/longs.")

    return parser.parse_args(args)
    return

if __name__ == "__main__":
    #main(**vars(setup()))
    response = main()
