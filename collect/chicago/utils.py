import json
import pandas as pd

def merge_inspections_reviews(inspections_path, reponse_json):

    inspecs = pd.read_parquet('data/inspections.parquet')
    inspecs['Address'] = inspecs['Address'].astype(str).str.strip()

    with open(reponse_json, "r") as rawfile:
            data = json.load(rawfile)

    inspected = {}

    for coord, resps in data.items():
        inspected[coord] = []
        for resp in resps:
            address = resp['location']['display_address'][0].upper()
            if len(inspecs[inspecs['Address'] == address]) > 0:
                inspected[coord].append(resp)

    
    

def coords_to_points(points_json):
    with open(points_json, "r") as f:
        points = json.load(f)

    points = points["points"]

    points_map = {}
    for idx, coord in enumerate(points):
        points_map[idx] = f"{coord['latitude']} , {coord['longitude']}"

    return points_map


def tag_resturant_json(rests: dict):
    tagged = {}
    for coord, resp in rests.items():
        part_tag = {}
        for row in resp:
            name = row["alias"]
            part_tag[name] = row

        tagged[coord] = part_tag

    return tagged


def create_scrape_list(rests: dict):
    check_list = {}
    for resp in rests.values():
        for rest_name in resp:
            check_list[rest_name] = False

    return check_list

