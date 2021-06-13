import pandas as pd
import json
import csv
import ssl 
import requests

def GetData_Smush():
    ## get data from cryptodownload 
    data_location = 'https://www.cryptodatadownload.com/cdd/Bitstamp_BTCUSD_minute.csv'
    ssl._create_default_https_context = ssl._create_unverified_context
    hist_data = pd.read_csv(data_location, skiprows=1)

    ## get latest data from bitstamp public API
    json_location=f'https://www.bitstamp.net/api/v2/ohlc/btcusd/?limit=1000&step=60'
    response =  requests.get(json_location)
    ld = json.loads(response.text)
    df = ld['data']['ohlc']
    latest_data = pd.DataFrame(df, columns=['high', 'timestamp', 'volume', 'low', 'close', 'open']) 

    latest_data['date'] = pd.to_datetime(latest_data['timestamp'], unit='s')
    latest_data['volume'] = pd.to_numeric(latest_data['volume'], errors='coerce')
    latest_data['close'] = pd.to_numeric(latest_data['close'], errors='coerce')
    latest_data['Volume_Currency'] = latest_data['volume']*latest_data['close'] 

    latest_data.columns = ['High', 'unix', 'Volume_BTC', 'Low', 'Close', 'Open', 'date', 'Volume_Currency']

            
    # re-arrange data
    dp1 = hist_data.head(20160)
    dp1 = dp1.drop('symbol', axis = 1)
    dp1.columns = ['unix','date','Open','High','Low','Close','Volume_BTC','Volume_Currency']
    row1 = latest_data.head(100)


    detail = int(row1.iloc[0]['unix'])
    dat = dp1.loc[dp1['unix'] < detail]
    dat = dat.iloc[::-1]
    latest_data = latest_data[['unix','date','Open','High','Low', 'Close', 'Volume_BTC', 'Volume_Currency']]



    dat= dat.append(latest_data)
    dat = dat.tail(20160)
    dat['High'] = pd.to_numeric(dat['High'], errors='coerce')
    dat['Low'] = pd.to_numeric(dat['Low'], errors='coerce')
    dat['Weighted_Price'] = (dat['High']+dat['Low'])/2
    
    dat = dat[['unix','date','Open','High','Low', 'Close', 'Weighted_Price','Volume_BTC', 'Volume_Currency']]

    ## return data
    return dat
  

smh = GetData_Smush()
print(smh)
