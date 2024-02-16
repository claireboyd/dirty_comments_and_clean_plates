# how_the_bear_got_a_C
Understanding how The Bear got a C: Using Restaurant Reviews to Predict Failed Restaurant Inspections in Chicago

**Determining the points to pull data for**

To replicate how we pulled random points to pull data from, run the below line of code:

```bash
poetry run python collect/identify_points.py --n 25 --output_filename points
```

**Getting restaurant data**

To replicate how we pulled in the restaurant data, you can run the below line of code:

```bash
poetry run python collect/get_restaurants_by_point.py --points_filepath data/test_points.json --output_filename test_restuarant_pull
```


