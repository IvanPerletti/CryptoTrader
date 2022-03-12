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
url = 'https://pro-api.coinmarketcap.com/v1/exchange/quotes/historical/20210101'
# url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/historical/20210101/'
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
  



import time
import requests


def get_timestamp(datetime: str):
    return int(time.mktime(time.strptime(datetime, '%Y-%m-%d %H:%M:%S')))

def getCryptoQuotes(symbol: str, start_date: str, end_date: str):
    start = get_timestamp(start_date)
    end = get_timestamp(end_date)
    url = f'https://web-api.coinmarketcap.com/v1/cryptocurrency/ohlcv/historical?symbol={symbol}&convert=USD&time_start={start}&time_end={end}'
    return requests.get(url).json()


crypto = [ 'ETH', 'BNB' , 'XRP' ]

for symbol in crypto:
    data = getCryptoQuotes(symbol,
                      start_date='2021-11-11 00:00:00',
                      end_date='2022-02-02 00:00:00')

    # making A LOT of assumptions here, hopefully the keys don't change in the future
    data_flat = [quote['quote']['USD'] for quote in data['data']['quotes']]
    df = pd.DataFrame(data_flat)
    
    print(df)


