from zipfile import ZipFile
import pandas as pd
import os
from os import listdir

def getData(frequency, file: str = "may_premium_dataset_BTC.zip"):
    allData = {}
    
    with ZipFile(file, "r") as zip_ref:
       # Get list of files names in zip
       list_of_files = zip_ref.namelist()
    
       # Iterate over the list of file names in given list & print them
       for elem in list_of_files:
           pairs = os.path.splitext(elem)[0]
           with zip_ref.open(pairs+'.csv') as f:
               df = pd.read_csv(f)
               df.drop(columns = ['Unnamed: 0', 'close_time', 'number_trades', 'asset'], inplace = True)
               
               df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
               df['time'] = pd.to_datetime(df['time'])
               
               new_df = pd.DataFrame()
               new_df['open'] = df.groupby(pd.Grouper(key = 'time', freq = frequency))['open'].first()
               new_df['high'] = df.groupby(pd.Grouper(key = 'time', freq = frequency))['high'].max()
               new_df['low'] = df.groupby(pd.Grouper(key = 'time', freq = frequency))['low'].min()
               new_df['close'] = df.groupby(pd.Grouper(key = 'time', freq = frequency))['close'].last()
               new_df['volume'] = df.groupby(pd.Grouper(key = 'time', freq = frequency))['volume'].sum()
               
               allData[pairs] = new_df
    
    idx = allData.values()[0].index
    
    for pairs, df in allData.items():
        idx = idx.intersection(df.index)
    
    for pairs, df in allData.items():
        df = df.loc[idx]
        path = 'RawData/' + pairs + '.csv'
        df.to_csv(path_or_buf = path, encoding='utf-8', index = True)
        print('success', pairs)

if __name__ == '__main__':
    getData('1D')
    