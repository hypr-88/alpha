from zipfile import ZipFile
import pandas as pd
from datetime import datetime
import os

data = dict()
symbolList = []


def importData(startTime: datetime, endTime: datetime, frequency: str = '1D', file: str = "may_premium_dataset_USDT.zip"):
    with ZipFile(file, "r") as zip_ref:
        # Get list of files names in zip
        list_of_files = zip_ref.namelist()

        # Iterate over the list of file names in given list
        for elem in list_of_files:
            # get the symbol
            symbol = os.path.splitext(elem)[0]

            # for symbol in main_pairs:
            # read csv
            with zip_ref.open(symbol + '.csv') as f:
                df = pd.read_csv(f)
                # drop columns
                df.drop(columns=['Unnamed: 0', 'close_time', 'number_trades', 'asset'], inplace=True)

                # rename the rest columns
                df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                # change time: str -> datetime
                df['time'] = pd.to_datetime(df['time'])

                # grouped data to with frequency
                new_df = pd.DataFrame()
                new_df['open'] = df.groupby(pd.Grouper(key='time', freq=frequency))['open'].first()
                new_df['high'] = df.groupby(pd.Grouper(key='time', freq=frequency))['high'].max()
                new_df['low'] = df.groupby(pd.Grouper(key='time', freq=frequency))['low'].min()
                new_df['close'] = df.groupby(pd.Grouper(key='time', freq=frequency))['close'].last()
                new_df['volume'] = df.groupby(pd.Grouper(key='time', freq=frequency))['volume'].sum()

                # drop nan
                new_df.dropna(inplace=True)

                # filter data have start date before and end date on
                if new_df.index[-1] == endTime and new_df.index[0] < startTime:
                    data[symbol] = new_df

    # get intersection trading date of all symbols
    idx = list(data.values())[0].index
    for symbol, df in data.items():
        idx = idx.intersection(df.index)

    for symbol, df in data.items():
        data[symbol] = df.loc[idx]
        print('success', symbol)


if __name__ == '__main__':
    importData(datetime(2019, 1, 1), datetime(2021, 5, 31))
    symbolList = list(data.keys())
