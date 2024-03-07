import requests
import lxml.html
from typing import List, Tuple
from datatypes import ReviewText
import jsonlines
import json
import time
import logging

logging.basicConfig(
    filename="error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

def scrape_single_page(url: str) -> Tuple[List[dict[str, str]], int]:
    """
    Scrapes single Yelp page

    Returns list of dictionaries, containing review text, user name, review date
    """
    s = requests.Session()
    resp = s.get(url)
    review_page = lxml.html.fromstring(resp.text)

    try:
        review_content = review_page.cssselect('[aria-label="Recommended Reviews"]')[0]
    except IndexError:
        logging.error(
            "Unable to scrape url: %s, Status Code: %s", url, resp.status_code
        )
        return None, resp.status_code

    review_text = review_content.cssselect("[class*=comment]")
    dates = review_content.cssselect('[class=" css-10n911v"]')
    users = review_content.cssselect('[class*="user-passport-info"] a[role="link"]')

    num_reviews = min(len(review_text), len(dates), len(users))

    page_data = []
    for i in range(min(num_reviews, 10)):
        review_data = {}
        review_data["text"] = review_text[i].text_content()
        review_data["date"] = dates[i].text_content()
        review_data["user"] = users[i].text_content()

        page_data.append(review_data)

    return page_data, resp.status_code


def parse_single_response(
    response: dict, filepath: str, max_reviews: int, clean=False, tokenize=False
) -> None:
    """
    Parses single API response, reads out JSONL files to specified file path
    """
    alias = response["alias"]
    for page_num in range(0, max_reviews, 10):
        url = f"https://www.yelp.com/biz/{alias}?start={page_num}#reviews"
        print(f"Scraping: {alias}, Review: {page_num} out of {max_reviews}")
        time.sleep(3)
        results, status = scrape_single_page(url)
        if not results:
            break
        aggregate_and_save(results, response, filepath, clean, tokenize)

    return status


def aggregate_and_save(
    results: dict, response: dict, filepath: str, clean: bool, tokenize: bool
):
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


def scrape(
    data: dict,
    out_folder: str,
    point: int,
    coords: dict,
    num_reviews: int,
    checklist: dict,
    override: int = float("inf"),
):
    """
    Scrapes all resturants based on inputed coordinates.
    """
    coord = coords[point]
    responses = data[coord]

    out_path = f"{out_folder}/zone_{point}.jsonl"
    for idx, response in enumerate(responses):
        rest_alias = response["alias"]
        if checklist[rest_alias]:
            continue
        status = parse_single_response(response, out_path, num_reviews)
        checklist[rest_alias] = True

        if status > 200 or idx == override:
            # save progress and break code
            with open(f"{out_folder}/checklist.json", "w") as f:
                json.dump(checklist, f)

            raise ValueError(f"Status code: {status} not valid")
