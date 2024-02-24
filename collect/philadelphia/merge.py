import pandas as pd
import json
from datetime import timedelta


# join design
def extract_relevant_reviews(row):
    # filter all reviews down to only those relevant for the inspection
    relevant_reviews = reviews_for_inspected_restaurants.loc[
        (reviews_for_inspected_restaurants.loc[:, "business_id"] == row["business_id"])
        & (
            reviews_for_inspected_restaurants.loc[:, "date"]
            >= row["prev_date_with_nulls"]
        )
        & (reviews_for_inspected_restaurants.loc[:, "date"] <= row["Inspection Date"]),
        :,
    ]
    # return both the list of text reviews and the stars given (if needed)
    n_reviews = len(list(relevant_reviews["stars"]))

    if len(list(relevant_reviews["stars"])) > 0:
        avg_rating = sum(list(relevant_reviews["stars"])) / len(
            list(relevant_reviews["stars"])
        )
    else:
        avg_rating = None

    return (
        list(relevant_reviews["text"]),
        list(relevant_reviews["stars"]),
        n_reviews,
        avg_rating,
    )


# read in bussiness json, subset to only PA
rest_data = []
with open("data/yelp_dataset/yelp_academic_dataset_business.json", "r") as data_file:
    for line in data_file:
        rest_data.append(json.loads(line))

rest_df = pd.DataFrame(rest_data)
rest_df
pa_restuarants = rest_df[rest_df["state"] == "PA"]


# inspections data
# downloaded from here: https://data.pa.gov/Public-Safety/Public-Food-Inspections-last-24-months-County-Agri/etb6-jzdg/about_data
phila_inspecs = pd.read_csv("data/phila/phila_inspections.csv")

# drop rows with missing labels
inspections = phila_inspecs.dropna(subset=["Overall Compliance"])
inspections

# pre-merge cleaning
inspections.loc[:, "Address"] = inspections.loc[:, "Address"].astype(str)
inspections.loc[:, "Address"] = inspections.loc[:, "Address"].str.strip()
inspections.loc[:, "Address"] = inspections.loc[:, "Address"].str.upper()

pa_restuarants.loc[:, "address"] = pa_restuarants.loc[:, "address"].astype(str)
pa_restuarants.loc[:, "address"] = pa_restuarants.loc[:, "address"].str.strip()
pa_restuarants.loc[:, "address"] = pa_restuarants.loc[:, "address"].str.upper()

labeled_inspections = pd.merge(
    inspections, pa_restuarants, how="left", left_on="Address", right_on="address"
).dropna()
labeled_inspections["Overall Compliance"].value_counts()


# get lists of resturant ids that have been inspected
inspected_restaurant_ids = list(
    labeled_inspections.loc[
        labeled_inspections["Overall Compliance"].notna(), "business_id"
    ]
)

# read in review text data
review_data = []
with open("data/yelp_dataset/yelp_academic_dataset_review.json", "r") as data_file:
    for line in data_file:
        review_data.append(json.loads(line))
review_df = pd.DataFrame(review_data)

# only keep reviews that are for restaurants in our inspected dataset
reviews_for_inspected_restaurants = review_df[
    review_df.loc[:, "business_id"].isin(inspected_restaurant_ids)
]

# convert date to datetime
reviews_for_inspected_restaurants.loc[:, "date"] = pd.to_datetime(
    reviews_for_inspected_restaurants.loc[:, "date"]
)


# get min and max dates in inspection data
labeled_inspections.loc[:, "Inspection Date"] = pd.to_datetime(
    labeled_inspections.loc[:, "Inspection Date"]
)
min_labeled_inspections_date = min(labeled_inspections.loc[:, "Inspection Date"])
max_labeled_inspections_date = max(labeled_inspections.loc[:, "Inspection Date"])

# subset reviews for only the ones within viable inspection periods
reviews_for_inspected_restaurants = reviews_for_inspected_restaurants.loc[
    (
        reviews_for_inspected_restaurants.loc[:, "date"]
        >= (min_labeled_inspections_date - timedelta(days=12 * 30))
    )
    & (
        reviews_for_inspected_restaurants.loc[:, "date"] <= max_labeled_inspections_date
    ),
    :,
]


labeled_inspections = labeled_inspections.sort_values(
    by=["Public Facility Name", "Inspection Date"], ascending=[True, True]
)
labeled_inspections["prev_date"] = labeled_inspections.groupby(
    ["Public Facility Name"]
)["Inspection Date"].shift()

# group all reviews within an inspection period (last inspection date or 6 months before if no last inspection)
labeled_inspections["prev_date_with_nulls"] = labeled_inspections["prev_date"]
labeled_inspections.loc[
    labeled_inspections["prev_date"].isna(), "prev_date_with_nulls"
] = labeled_inspections["Inspection Date"] - timedelta(12 * 30)


reviews_for_inspected_restaurants  # .groupby("business_id").count()


extract_relevant_reviews(labeled_inspections.iloc[0, :])

# apply the above function to each row of the dataset
sorted_reviews = labeled_inspections.apply(
    extract_relevant_reviews, axis=1, result_type="expand"
).rename(columns={0: "reviews", 1: "ratings", 2: "n_reviews", 3: "avg_rating"})
labeled_inspections_with_reviews = labeled_inspections.merge(
    right=sorted_reviews, right_index=True, left_index=True
)

# filter out observations without reviews
labeled_inspections_with_reviews = labeled_inspections_with_reviews.loc[
    labeled_inspections_with_reviews.loc[:, "n_reviews"] > 0, :
]
labeled_inspections_with_reviews.to_csv(
    "data/phila/labeled_inspections_with_reviews.csv"
)
