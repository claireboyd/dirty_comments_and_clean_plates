# how_the_bear_got_a_C
Understanding how The Bear got a C: Using Restaurant Reviews to Predict Failed Restaurant Inspections in Chicago

**Determining the points to pull data for**

To replicate how we pulled random points to pull data from, run the below line of code:

```bash
poetry run python collect/identify_points.py --n 20 --output_filename data/points.json
```

**Getting restaurant data**

To replicate how we pulled in the restaurant data, you can run the below line of code:

```bash
poetry run python collect/get_restaurants_by_point.py --points_filepath data/points.json --output_filename data/restuarant_pull.json
```

**Cleaning inspections data**

To replicate this part of the process, run the following line of code:

```bash
poetry run python collect/get_inspections.py --inspections_fp data/inspections.parquet --restaurants_fp data/test_restaurant_pull.json
```
