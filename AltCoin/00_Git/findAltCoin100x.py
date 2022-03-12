# -*- coding: utf-8 -*-
"""
Created on Sun Feb  6 07:06:52 2022

@author: perletti
"""
# https://medium.datadriveninvestor.com/finding-the-100x-altcoin-3c9af2bdd7b5


from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import pandas as pd
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
# Retrieve the top 5000 coins based on market cap

url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
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
  
coindf = pd.json_normalize(data['data'])
coindf2 = coindf[ (coindf['quote.USD.market_cap'] < 10000000) & (coindf['quote.USD.market_cap'] > 5000000)]



coindf3 = coindf2[ (coindf2['quote.USD.volume_24h']/coindf2['quote.USD.market_cap']).between(0.1,0.5)   ]
coindf3
  
  # filter out coins not trading majority on trustworth exchanges

from pycoingecko import CoinGeckoAPI
cg = CoinGeckoAPI()

crypto_list_shortlisted = []

for index, row in coindf3.iterrows():
    symbol1 = row.symbol.lower()
    symbol2 = coindf3.loc[index]['name'].lower()
    
    try:
        print(symbol1)
        coin_exchange = cg.get_coin_by_id(id = symbol1 )['tickers']
        df = pd.DataFrame(coin_exchange)
        df.trust_score.value_counts()
    except:
        print(symbol2)
        try:
            coin_exchange = cg.get_coin_by_id(id = symbol2 )['tickers']
        except:
            continue
        df = pd.DataFrame(coin_exchange)
        df.trust_score.value_counts()
    try :
        if df.groupby('trust_score')['volume'].sum()['green']/df.groupby('trust_score')['volume'].sum().sum() >= 0.5:
            crypto_list_shortlisted.append(symbol1)
    except:
        continue

  
coindf4 = coindf3[coindf3.symbol.str.lower().isin(crypto_list_shortlisted)]
  
coindf4
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  