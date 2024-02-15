# how_the_bear_got_a_C
Understanding how The Bear got a C: Using Restaurant Reviews to Predict Failed Restaurant Inspections in Chicago

**Getting restaurant data**

To replicate how we pulled in the restaurant data, you can use the below line of code:

```bash
poetry run python collect/get_restaurants.py --points_filepath collect/test_points.json --output_filename test_restuarant_pull
```

**Determining the points to pull data for**

To replicate how we pulled random points to pull data from, 

