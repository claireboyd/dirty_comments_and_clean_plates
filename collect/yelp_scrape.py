import requests
import lxml.html
from typing import List


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


def paginate_reviews(resturant_name: str, max_reivews: int) -> List[dict[str]]:
    """
    Paginates through multiple pages of reviews
    """
    all_reviews = []
    for page_num in range(0, max_reivews, 10):
        url = f"https://www.yelp.com/biz/{resturant_name}?start={page_num}#reviews"
        results = scrape_single_page(url)
        if not results:
            break
        all_reviews += results

    return all_reviews
