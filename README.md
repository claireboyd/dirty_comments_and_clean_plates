# Dirty Comments, Clean Plates
Dirty Comments, Clean Plates: Using Restaurant Reviews to Predict Failed Restaurant Inspections in Philadelphia

Our task: use a corpus of text-based reviews to train a model to classify if a restaurant is likely to fail a health inspection

Our project has had the following phases:
* attempting to use Chicago inspections and Yelp review data
* pivoting to a philadelphia-based approach when our scraping efforts were throttled
* creating NN and transformer-based models
* building a fake reviews dataset
* applying our binary-classification models on the fake reviews labeled dataset

This repository has the following directories, which include code for how we constructed our project:
* `collect/`: contains two sub-directories that chronicle our data collection efforts for both the Chicago-based approach and the philadelpha approach.
* `data/`:
* `eda/`:
* `models/`:
* `chat_gpt/`:











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
