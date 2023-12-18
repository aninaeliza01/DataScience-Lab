import requests

response=requests.get('https://www.ajce.in/')
if response.status_code==200:
    print("content\n",response.text)
else:
    print("failed",response.status_code)

