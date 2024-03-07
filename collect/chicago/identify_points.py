import argparse, geodatasets, shapely, json
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

def get_n_geopoints(n):
    """
    Uses chicago shapefile and spatial joins to generate a GeoSeries of 
    n points within the chicago boundaries.
    """
    #combine all 77 neighborhoods into one big polygon
    chicago = gpd.read_file(geodatasets.get_path("geoda.chicago_commpop"))
    neighborhood_polygons=chicago['geometry']
    chicago_polygon = gpd.GeoDataFrame(index=["myPoly"], geometry=[shapely.ops.unary_union(neighborhood_polygons)])

    # https://gis.stackexchange.com/questions/294394/randomly-sample-from-geopandas-dataframe-in-python
    # find the bounds of your geodataframe
    x_min, y_min, x_max, y_max = chicago_polygon.total_bounds

    # generate random data within the bounds
    x = np.random.uniform(x_min, x_max, 150)
    y = np.random.uniform(y_min, y_max, 150)

    # convert them to a points GeoSeries
    gdf_points = gpd.GeoSeries(gpd.points_from_xy(x, y))
    # only keep those points within the chicago polygon
    gdf_points = gdf_points[gdf_points.within(chicago_polygon.unary_union)]
    return gdf_points[:n]

def save_as_json(gdf_points, output_filename):
    """
    Convert the GeoSeries into an output which get_restaurants can use.
    """
    output={}
    output["points"]=[]

    for point in gdf_points.geometry:
        result = {}
        result['latitude'] = point.y
        result['longitude'] = point.x
        output["points"].append(result)

    with open(f"{output_filename}", "w") as outfile:
        json.dump(output, outfile)

def main(n: str, output_filename: str):
    # run both functions and save to data folder
    gdf_points = get_n_geopoints(int(n))
    save_as_json(gdf_points, output_filename)

def setup(args=None):    
    parser = argparse.ArgumentParser(description='Gets random n lat/long points in chicago.')
    parser.add_argument('--n', required=True, type=str, dest="n", help="Number of random points to pull within Chicago.")
    parser.add_argument('--output_filename', required=True, type=str, dest="output_filename", help="Filename.")
    return parser.parse_args(args)

if __name__ == "__main__":
    main(**vars(setup()))
