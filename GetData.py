import requests
import pandas as pd
from datetime import datetime

def getData(fromDate: tuple = (2016, 1, 1), toDate: tuple = 'now'):
    '''
    This function request data of companies in 'RawSP500.csv' from TD Ameritrade using my personal user ID 
    and save data to 'RawData/{Symbol}.csv'
    
    Parameters
    ----------
    fromDate : tuple, optional
        (YYYY, MM, DD). The default is (2016, 1, 1).
    toDate : tuple, optional
        (YYYY, MM, DD). The default is 'now'.

    Returns
    -------
    None.

    '''
    SP500 = pd.read_csv('RawSP500.csv')
    
    for i in range(len(SP500)):
        symbol = SP500['Symbol'][i]
        
        # Source
        URL = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(symbol)
    
        # api_ID
        api_ID = 'AXYGAOVL7KNBURXFL9RGNGAJGGOY02TZ'
    
        # set epoch date
        start_date = str(round(datetime(fromDate[0], fromDate[1], fromDate[2]).timestamp()*1000))
        if toDate == 'now':
            end_date = str(round(datetime.now().timestamp()*1000)) # now
        else:
            end_date = str(round(datetime(toDate[0], toDate[1], toDate[2]).timestamp()*1000))
    
        # payload
        payLoad = {'apikey':api_ID,
                   'periodType': 'year',
                       #'day','month','year','ytd'
                   'frequencyType': 'daily',
                       #'minute','daily','weekly','monthly'
                   'period':'10',
                   'frequency':'1',
                   'startDate':start_date,
                   'endDate':end_date,
                   'needExtendedHoursData': 'false'}
                       #'true', 'false'
    
        # Request data
        data = requests.get(url = URL, params = payLoad).json()
        if data == {'error': 'Not Found'}:
            print('Cannot request API', symbol)
            SP500.drop(i, inplace = True)
            continue
        
        # Convert data from dictionary to DataFrame
        data = pd.DataFrame(data['candles'])
        
        if symbol == 'MMM': shape = data.shape
        if symbol == 'MMM' or data.shape == shape and i < 100: #comment i<100 to get data for all comapy in SP500
            # convert epoch to date
            data['datetime'] = pd.to_datetime(data['datetime'], unit = 'ms')
    
            #save file
            path = 'RawData/' + symbol + '.csv'
            
            print('success', symbol)
            # save data
            data.to_csv(path_or_buf = path, encoding='utf-8', index = False)
        else:
            print('fail', symbol, data.shape)
            SP500.drop(i, inplace = True)
    
    SP500.to_csv(path_or_buf = 'SP500.csv', encoding='utf-8', index = False)

if __name__ == '__main__':
    getData()