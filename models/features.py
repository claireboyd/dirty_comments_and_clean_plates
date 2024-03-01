import pandas as pd
import geopandas as gpd
import argparse

REL_COLS = [
    "Active Indicator",
    "Public Facility Name",
    "Address",
    "City",
    "County Name",
    "Zip Code",
    "State",
    "Inspection Date",
    "Inspection Reason Type",
    "Overall Compliance",
    "business_id",
    "name",
    "latitude",
    "longitude",
    "stars",
    "review_count",
    "is_open",
    "categories",
    "prev_date",
    "prev_date_with_nulls",
    "reviews",
    "ratings",
    "n_reviews",
    "avg_rating",
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

    for cat in CATEGORIES:
        df[cat] = 0
        df.loc[df["categories"].str.contains(cat), cat] = 1

        if df[cat].sum() < n:
            df = df.drop(columns=cat)

    return df.drop(columns=["categories"])


def categorize_population(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features based on population.
    """
    pa_counties = gpd.read_file("data/phila/pa_county")
    df_w_pop = pd.merge(
        df,
        pa_counties[["NAME20", "TOTPOP20"]],
        left_on="County Name",
        right_on="NAME20",
    )

    df_w_pop["below_500k"] = 0
    df_w_pop["above_500k"] = 0

    df_w_pop.loc[df_w_pop["TOTPOP20"] >= 500000, "below_500k"] = 1
    df_w_pop.loc[df_w_pop["TOTPOP20"] < 500000, "above_500k"] = 1

    return df_w_pop.drop(columns=["TOTPOP20", "NAME20"])


def main(file_name: str):
    phila = pd.read_csv("data/phila/labeled_inspections_with_reviews.csv")
    phila = phila[REL_COLS]
    encoded = encode_categories(phila, 25)
    encoded = categorize_population(encoded)
    encoded.to_csv(f"data/phila/{file_name}", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=True)
    args = parser.parse_args()
    main(args.file_name)
