import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import urllib
from urllib.request import urlretrieve
import pandas as pd

from time import sleep

# Format changed over the years 
# 96-99 use p's 
# 00-01 use li's 
# 02-now use 

# Base URL
base_url = "https://psb.stanford.edu/psb-online/proceedings/psb"


# Loop through each year from 1996 to 2024
for year in range(2024, 2025):
    d = {'pdf':[],'authors': [], 'titles': [], 'number': [], 'available':[]}
    df = pd.DataFrame(data=d)
    year_suffix = str(year)[-2:]
    url = f"{base_url}{year_suffix}/"

    response = requests.get(url, verify=False)
    if response.status_code != 200:
        print(f"Failed to retrieve data for year {year}.")
        continue    

    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract PDF links
    pdf_links = soup.find_all('dt')
    pdf_all = []
    dl = soup.find_all('dd')
    print(len(pdf_links), len(dl))
    for i, dt in enumerate(pdf_links):
        link = dt.select_one('a')
        if link and link['href'].endswith('.pdf') and len(link['href'])>4:
            title = ' '.join([val.strip() for val in link.text.strip().split('\n')])
            if title.lower() == "session introduction" or title.lower() == "preface" or title.lower() == "introduction":
                continue
            if dl[i].select_one('i'):
                authors = ' '.join([val.strip() for val in dl[i].select_one('i').text.strip().split('\n')])
            else:
                authors = ' '.join([val.strip() for val in dl[i].text.strip().split('\n')])
            pdf_all.append({'pdf':urljoin(url, link['href']), 'authors': authors, 'titles':title})
            

    # pdf_links = [urljoin(url, link['href']) for link in pdf_links if link['href'].endswith('.pdf')]
    # Create a folder to save PDFs
    pdf_folder = f'PSB_Papers/PDFS/psb{year}_pdfs'
    os.makedirs(pdf_folder, exist_ok=True)
    # Download PDFs
    for i in range(len(pdf_all)):
        # response = requests.get(pdf_link)
        pdf_all[i]['number']=i
        pdf_link = pdf_all[i]['pdf']
        pdf_link = pdf_link.replace(" ", "%20")
        print("wassup", pdf_link)
        pdf_response = requests.get(pdf_link, verify=False)
        try:
            urlretrieve(pdf_link)
        except Exception as err:
            pdf_all[i]['available']=False
            df2 = pd.DataFrame(data=[pdf_all[i]])
            df= pd.concat([df, df2], ignore_index=True)
            print(f"Other error occurred: {err}")
        else:
            pdf_all[i]['available']=True
            df2 = pd.DataFrame(data=[pdf_all[i]])
            print(df2)
            df = pd.concat([df, df2], ignore_index=True)
            urlretrieve(pdf_link, f"{pdf_folder}/{i}")
        df['available'] = df['available'].astype(bool)
        df.to_csv(f"PSB_Papers/CSVs/{year}.csv")
        print(f'Downloaded: {i}')
    sleep(10)
