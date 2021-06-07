import pandas as pd

def EMA(series: pd.Series, window: int):
    return series.ewm(window).mean()

def ESTD(series: pd.Series, window: int):
    return  series.ewm(window).std()

def SMA(series: pd.Series, window: int):
    return series.rolling(window).mean()

def STD(series: pd.Series, window: int):
    return series.rolling(window).std()