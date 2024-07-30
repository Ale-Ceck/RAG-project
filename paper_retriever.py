import pandas as pd
import arxiv
import os
import re
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
CSV_PATH = os.getenv("CSV_PATH")
DATA_PATH = os.getenv("DATA_PATH")

df = pd.read_csv(f'{CSV_PATH}arxiv_dataset.csv')
df.drop(df[df['Open Access Link'].isna()].index, inplace=True)
df = df.drop(index=0)
df_arxiv = df[df['Open Access Link'].str.contains('arxiv')]

def main(): 
    print(df_arxiv.head())
    for index, document in df_arxiv.iterrows():
        url = extract_url(document[2])
        article_id = parse_id(url)
        print(f' \n article_id')
        arxiv_download(article_id, DATA_PATH)
    print("Download completed.")
    

def arxiv_download(article_id, download_path):
    os.makedirs(download_path, exist_ok=True)
    search_paper = arxiv.Client().results(arxiv.Search(id_list=[article_id]))

    if article := next(search_paper):
        print(f'Starting download of article: "{article.title}" ({article.entry_id})')
        saving_path = article.download_pdf(dirpath=download_path)
        print(f'Downloaded at: \n{saving_path}')
    else:
        print(f"Article ({article_id}) not found.")

def extract_url(open_access_url):
    url_pattern = re.compile(r"\{'url': '(https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?)', 'status': '[^']+'\}")
    match = url_pattern.search(open_access_url)
    if match:
        return match.group(1)
    else: 
    # If the input does not match any of the expected formats
        print(open_access_url)
        print("Error: The provided input does not match the expected format.")
        exit(1)

def parse_id(article_link):
    # Pattern to match an arXiv URL and extract the ID
    url_pattern = re.compile(r"https?://(?:www\.)?arxiv\.org/(abs|pdf)/(\d{4}\.\d{4,5})(v\d+)?(\.pdf)?$")
    url_match = url_pattern.match(article_link)
    if url_match:
        return url_match.group(2) + (url_match.group(3) if url_match.group(3) else "")

    # If the input does not match any of the expected formats
    print("Error: The provided input does not match the expected URL or ID format.")
    exit(1)


if __name__ == "__main__":
    main()