from bs4 import BeautifulSoup
import requests

url = 'https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers'
response = requests.get(url)

toret = []
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all <tr> elements which contain the paper titles
    papers = soup.find_all('tr')

    count = 1  # Counter for numbering titles

    for paper in papers:
        # Attempt to find <td> element
        td_element = paper.find('td')

        if td_element:
            # Attempt to find <strong> element within <td>
            title_element = td_element.find('strong')
            
            if title_element:
                title = title_element.text.strip()
                if 'diffusion' in title.lower():
                    print(f"{count}. {title}")
                    toret.append(f"{count}. {title}")
                    count += 1

            else:
                # If no <strong> tag, try finding <a> tag (assuming title is linked)
                a_element = td_element.find('a')
                if a_element:
                    title = a_element.text.strip()
                    if 'diffusion' in title.lower():
                        print(f"{count}. {title}")
                        toret.append(f"{count}. {title}")
                        count += 1
else:
    print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")

# Write collected titles with numbers to a text file
with open('output.txt', 'w', encoding='utf-8') as file:
    for title in toret:
        file.write(title + '\n')

print(f"Successfully wrote {len(toret)} titles to output.txt.")
