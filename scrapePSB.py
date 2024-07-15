import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

#asdfasdf
# Base URL
base_url = "https://psb.stanford.edu/psb-online/proceedings/psb"

# Loop through each year from 1996 to 2024
for year in range(1996, 2025):
    # Convert year to the appropriate format
    year_suffix = str(year)[-2:]
    url = f"{base_url}{year_suffix}/"
    print(year_suffix)
    if year_suffix != '16':
        continue
    # Fetch the HTML content from the website
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to retrieve data for year {year}.")
        continue
    html_content = response.text

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    # Extract PDF links
    pdf_links = []
    titles = []

    for dt in soup.find_all('dt'):
        link = dt.find('a')
        if link and link['href'].endswith('.pdf'):
            pdf_links.append(urljoin(url, link['href']))
            titles.append(link.text.strip())

    with open('titles.txt', 'w', encoding='utf-8') as f:
        for title in titles:
            f.write(title + '\n')
    # Create a folder to save PDFs
    pdf_folder = f'psb{year}_pdfs'
    os.makedirs(pdf_folder, exist_ok=True)
    print(titles)
    # Download PDFs
    for pdf_link in pdf_links:
        response = requests.get(pdf_link)
        pdf_name = os.path.join(pdf_folder, pdf_link.split('/')[-1])
       # with open(pdf_name, 'wb') as pdf_file:
       #    pdf_file.write(response.content)
        print(f'Downloaded: {pdf_name}')
