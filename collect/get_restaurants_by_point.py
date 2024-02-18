from pathlib import Path
import os, sys, json, requests, argparse
from dotenv import load_dotenv

#import vars from .env to os.environ
load_dotenv()
YELP_API_KEY = os.getenv('YELP_API_KEY')

def fetch_results(params, limit=50):
    """
    Uses a dictionary of params to send a search_query to the YELP Fusion API
    https://docs.developer.yelp.com/reference/v3_business_search

    Returns a requests response object.
    """
    try:
        #build query
        base_url="https://api.yelp.com/v3/businesses/search?"

        #provide necessary headers
        headers = {"accept": "application/json",
                   "Authorization": f"Bearer {YELP_API_KEY}"}
        
        params["limit"] = limit

        #get response and save
        response = requests.get(url=base_url, params=params, headers=headers)
        return response        

    except Exception as e:
        raise str(e)

def get_restaurants_for_points(points_filepath, output_filename):
    """
    Takes a filepath of a json object that has a key "points", with a value
    that includes a list of dictionaries with keys for "latitude" and
    "longitude" to get restaurant objects for.

    Returns: nothing. Saves a json object in the data/ dir with the output 
    filename that contains a dictionary with str key that concatenates
    the lat/long points, and the value of a list of dictionaries that are all 
    restaurant objects.
    """
    #initialize output dict/json object
    output = {}
    
    #load in params (list of points)
    with open(points_filepath) as json_data:
        points = json.load(json_data)

    for point_params in points["points"]:
        #create unique key for each lat/long combo
        key = str(point_params['latitude'])+" , "+str(point_params['longitude'])
        print(f"Starting with point: {key}")
              
        results = []
        for i in range(0, 1000, 50):
            print(f"Processing restaurant {i}...")
            point_params["offset"] = i

            #get response for specific params
            response = fetch_results(params=point_params)
            results.append(response.json()['businesses'])

        output[key] = [restaurant for batch in results for restaurant in batch]
        print(f"Done with point: {key}")

    with open(f"{output_filename}", "w") as outfile:
        json.dump(output, outfile)

def main(points_filepath: Path, output_filename: str):
    get_restaurants_for_points(points_filepath, output_filename)


def setup(args=None):    
    parser = argparse.ArgumentParser(description='Gets JSON of Yelp Businesses from list of dictionaries of Lat/Long values.')

    parser.add_argument('--points_filepath', required=True, type=Path, dest="points_filepath", help="Path to file with points for query.")
    parser.add_argument('--output_filename', required=True, type=Path, dest="output_filename", help="Filename.")
    return parser.parse_args(args)


if __name__ == "__main__":
    main(**vars(setup()))
