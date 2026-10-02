import requests
import pandas as pd
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin

data = []

for page in range(1, 26):

    url = f"https://books.toscrape.com/catalogue/page-{page}.html"

    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")

    books = soup.find_all("article", class_="product_pod")

    # Loop through every book on the page
    for book in books:

    
        title = book.h3.a["title"]

        price = book.find("p", class_="price_color").text

        rating = book.find("p", class_="star-rating")["class"][1]

        availability = book.find("p", class_="instock availability").text.strip()

        relative_link = book.h3.a["href"]

        product_url = urljoin(url, relative_link)

        product_response = requests.get(product_url)

        product_soup = BeautifulSoup(product_response.text, "html.parser")

        # UPC and Product Type
        table = product_soup.find("table")

        rows = table.find_all("tr")

        upc = rows[0].td.text

        product_type = rows[1].td.text

        # Category
        breadcrumb = product_soup.find("ul", class_="breadcrumb")

        category = breadcrumb.find_all("li")[2].text.strip()

        # Description
        description = ""

        desc = product_soup.find("div", id="product_description")

        if desc:
            description = desc.find_next("p").text.strip()

        # Store data
        item = {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Availability": availability,
            "Product URL": product_url,
            "Category": category,
            "UPC": upc,
            "Product Type": product_type,
            "Description": description
        }

        data.append(item)

# Save JSON
with open("books.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

# Convert to DataFrame
df = pd.DataFrame(data)

print("Total Books Scraped:", len(df))

# Save CSV
df.to_csv("books.csv", index=False, encoding="utf-8-sig")

print("JSON file saved successfully.")
print("CSV file saved successfully.")


