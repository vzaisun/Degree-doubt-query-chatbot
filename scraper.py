import requests
from bs4 import BeautifulSoup

URLS = [
    "https://datascience.ucsd.edu/current-students/course-offerings/",
    "https://datascience.ucsd.edu/graduate/ms-program/",
    "https://datascience.ucsd.edu/graduate/graduate-admissions/",
    "https://datascience.ucsd.edu/graduate/ms-program/progress-to-degree/",
    "https://mds.ucsd.edu/program/index.html",
]

def scrape():
    all_text = []

    for url in URLS:
        print(f"Scraping {url}")
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")

        # remove scripts/styles
        for tag in soup(["script", "style"]):
            tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
        all_text.append(text)

    with open("ucsd_docs.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_text))

    print("Saved ucsd_docs.txt")

if __name__ == "__main__":
    scrape()
