import pandas as pd
import geopandas as gpd
import argparse
from sklearn.model_selection import train_test_split
import os


FEATURES = [
    "Overall Compliance",
    "name",
    "stars",
    "review_count",
    "is_open",
    "reviews",
    "ratings",
    "n_reviews",
    "avg_rating",
    "IR_regular",
    "IR_follow_up",
    "IR_other",
    "Chester",
    "Bucks",
    "Philadelphia",
    "Delaware",
    "Montgomery",
    "Berks",
]

CATEGORIES = [
    "Nightlife",
    "Bars",
    "American (Traditional)",
    "Pizza",
    "American (New)",
    "Italian",
    "Sandwiches",
    "Breakfast & Brunch",
    "Cafes",
    "Burgers",
    "Delis",
    "Caterers",
    "Mexican",
    "Desserts",
    "Salad",
    "Sports Bars",
    "Pubs",
    "Chicken Wings",
    "Seafood",
    "Beer",
    "Wine & Spirits",
    "Juice Bars & Smoothies",
    "Mediterranean",
    "Gastropubs",
    "Diners",
    "Steakhouses",
    "Breweries",
    "Donuts",
    "Barbeque",
    "Buffets",
    "Gelato",
    "French",
    "Chicken Shop",
    "Tacos",
    "Beer Gardens",
    "Comfort Food",
    "Taiwanese",
    "Cheesesteaks",
    "Middle Eastern",
    "Wineries",
    "Indian",
    "Halal",
    "Vegan",
    "Vegetarian",
    "Thai",
    "Food Trucks",
    "Bagels",
    "Brewpubs",
    "Food Delivery Services",
    "Organic Stores",
    "Pakistani",
    "Shaved Ice",
    "Beer Bar",
    "Czech",
    "Falafel",
    "Hot Dogs",
    "Creperies",
    "Hot Pot",
    "Tapas Bars",
    "Acai Bowls",
    "Noodles",
    "Cupcakes",
    "Modern European",
    "Bubble Tea",
    "Vietnamese",
    "Soup",
    "Sushi Bars",
    "Dim Sum",
    "Ramen",
    "Tapas/Small Plates",
    "Creperies",
    "Fondue",
]


def encode_categories(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    One hot endcode by resturant type, only keep features with at least
    n observations.
    """
    df["categories"] = df["categories"].str.replace(r"\([^()]*\)", "", regex=True)
    df["categories"] = df["categories"].str.strip()
    kept_columns = []

    for cat in CATEGORIES:
        df[cat] = 0
        df.loc[df["categories"].str.contains(cat), cat] = 1

        if df[cat].sum() < n:
            df = df.drop(columns=cat)
        else:
            kept_columns.append(cat)

    return df.drop(columns=["categories"]), kept_columns


def categorize_counties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on population.
    """
    pa_counties = gpd.read_file("data/phila/pa_county").to_crs("EPSG:4326")
    geo_df = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    )
    df_counties = gpd.sjoin(
        geo_df,
        pa_counties[["NAME20", "TOTPOP20", "geometry"]],
        how="left",
        predicate="within",
    )

    for county in df_counties["NAME20"].unique():
        df_counties[county] = 0
        df_counties.loc[df_counties["NAME20"] == county, county] = 1

    return pd.DataFrame(df_counties)


def cat_inspection_reason(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encoding based on inpsection reason
    """
    df["IR_regular"] = 0
    df["IR_follow_up"] = 0
    df["IR_other"] = 0

    df.loc[df["Inspection Reason Type"] == "Regular", "IR_regular"] = 1
    df.loc[df["Inspection Reason Type"] == "Follow Up", "IR_follow_up"] = 1
    df.loc[(df["IR_regular"] != 1) & (df["IR_follow_up"] != 1), "IR_other"] = 0

    return df


def main(file_name: str, split: bool = False):
    phila = pd.read_csv("data/phila/labeled_inspections_with_reviews.csv")
    phila["Inspection Date"] = pd.to_datetime(phila["Inspection Date"])
    phila["Inspection Date"] = phila["Inspection Date"].dt.month
    phila = phila.rename(columns={"Inspection Date": "Month"})
    encoded, food_cats = encode_categories(phila, 25)
    encoded = categorize_counties(encoded)
    encoded = cat_inspection_reason(encoded)
    encoded = encoded[FEATURES + food_cats]

    if split:
        os.makedirs("data/split", exist_ok=True)
        encoded = encoded.reset_index().rename(columns={"index": "uuid"})
        val = encoded.sample(frac=0.10)
        leftover = encoded[~encoded["uuid"].isin(val["uuid"].to_list())]
        train, test = train_test_split(leftover, train_size=0.90, shuffle=True)

        val.drop(columns=["uuid"]).to_csv("data/split/val.csv", index=False)
        train.drop(columns=["uuid"]).to_csv("data/split/train.csv", index=False)
        test.drop(columns=["uuid"]).to_csv("data/split/test.csv", index=False)
    else:
        encoded.to_csv(f"data/{file_name}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=False)
    parser.add_argument("--split", action="store_true", required=False)
    args = parser.parse_args()
    main(args.file_name, args.split)
