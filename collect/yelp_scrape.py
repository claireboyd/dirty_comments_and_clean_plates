import requests
import lxml.html
from typing import List
from datatypes import ReviewText
import jsonlines
import json
import os


def scrape_single_page(url: str) -> List[dict[str, str]]:
    """
    Scrapes single Yelp page

    Returns list of dictionaries, containing review text, user name, review date
    """
    resp = requests.get(url)
    review_page = lxml.html.fromstring(resp.text)
    review_content = review_page.cssselect('[aria-label="Recommended Reviews"]')[0]

    review_text = review_content.cssselect("[class*=comment]")
    dates = review_content.cssselect('[class=" css-10n911v"]')
    users = review_content.cssselect('[class*="user-passport-info"] a[role="link"]')

    page_data = []
    for i in range(min(len(review_text), 10)):
        review_data = {}
        review_data["text"] = review_text[i].text_content()
        review_data["date"] = dates[i].text_content()
        review_data["user"] = users[i].text_content()

        page_data.append(review_data)

    return page_data


def parse_single_response(
    response: dict, filepath: str, max_reviews: int, clean=False, tokenize=False
) -> None:
    """
    Parses single API response, reads out JSONL files to specified file path
    """

    alias = response["alias"]
    for page_num in range(0, max_reviews, 10):
        url = f"https://www.yelp.com/biz/{alias}?start={page_num}#reviews"
        print(f'Scraping: {alias}, Review: {page_num} out of {max_reviews}')
        results = scrape_single_page(url)
        if not results:
            break
        aggregate_and_save(results, response, filepath, clean, tokenize)


def aggregate_and_save(results, response, filepath, clean, tokenize):
    """
    Aggergates text and API data, cleans and tokensize, and saves to specifiy file path
    """
    with jsonlines.open(filepath, "a") as out:

        for review in results:
            data = ReviewText(
                resturant_name=response["name"],
                alias=response["alias"],
                text=review["text"],
                date=review["date"],
                is_closed=response["is_closed"],
                address=response["location"]["display_address"][-1],
                rating=response["rating"],
            )

            if clean:
                data.clean()

            if tokenize:
                data.tokenize()

            out.write(data.model_dump(mode="json"))


def scrape(in_path, out_folder, coords, num_reviews, start=None, stop=None):
    with open(in_path, "r") as rawfile:
        data = json.load(rawfile)

    responses = data[coords]
    lat, _, lon = coords.split()

    out_path = f"{out_folder}/coordinates_{lat[:5]}_{lon[:5]}.jsonl"
    for response in responses[start:stop]:
        parse_single_response(response, out_path, num_reviews)


if __name__ == "__main__":
    test_coords = "41.755097245008066 , -87.63462521491509"
    scrape(
        "data/restuarant_pull.json",
        out_folder="data",
        coords=test_coords,
        num_reviews=30,
        stop=10,
    )
