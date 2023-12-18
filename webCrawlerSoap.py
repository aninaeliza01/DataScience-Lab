import requests
from bs4 import BeautifulSoup

response = requests.get("https://ajce.in")
if response.status_code == 200:
 soup=BeautifulSoup(response.content,'html.parser')
 print("Title:",soup.title.string)
 print("Content:")
 print(soup.get_text())
else:
    print("Failed to fetch the page. Status code:", response.status_code)
