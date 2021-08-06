import pandas as pd
import talib
from arch import arch_model

def EMA(series: pd.Series, window: int):
    return talib.EMA(series, timeperiod=window)


def ESTD(series: pd.Series, window: int):
    return series.ewm(window).std()


def SMA(series: pd.Series, window: int):
    return talib.SMA(series, timeperiod=window)


def STD(series: pd.Series, window: int):
    return talib.STDDEV(series, timeperiod=window)


def MACD(series: pd.Series, fastperiod: int, slowperiod: int, signalperiod: int):
    return talib.MACD(series, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)


def RSI(series: pd.Series, window: int):
    return talib.RSI(series, timeperiod=window)


def BBANDS(series: pd.Series, window: int, nbdevup: int, nbdevdn: int):
    return talib.BBANDS(series, timeperiod=window, nbdevup=nbdevup, nbdevdn=nbdevdn, matype=0)


def Volatility(returns: pd.Series):
    returns *= 100 # convert returns to %. For convergence purpose of the algorithm
    model = arch_model(returns, vol='GARCH', p=1, o=1, q=1, dist='normal')
    result = model.fit()
    return result.conditional_volatility