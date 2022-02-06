# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 08:04:05 2022

@author: perletti
"""

import csv  
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
# Retrieve the top 5000 coins based on market cap

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/historical/20210101/'
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': '9041b8bb-0c9b-450b-850e-56676e764b94',
}

session = Session()
session.headers.update(headers)

try:
  response = session.get(url, params=parameters)
  data = json.loads(response.text)
  # print(data)
except (ConnectionError, Timeout, TooManyRedirects) as e:
  print(e)
  






header = ['name', 'area', 'country_code2', 'country_code3']
data = ['Afghanistan', 652090, 'AF', 'AFG']

with open('countries.csv', 'w', encoding='UTF8',newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    writer.writerow(coindf)

    
# -*- coding: utf-8 -*-