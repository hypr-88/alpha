import numpy as np
import pandas as pd
from enum import Enum

from sklearn.preprocessing import MinMaxScaler
from typing import Dict

from Features import EMA, ESTD, STD, SMA, RSI, MACD, BBANDS, Volatility

basicFeatures = ['open', 'close', 'high', 'volume', 'return']


class DeriveFeature(Enum):
    EMA = 'EMA'
    STD = 'STD'
    ESTD = 'ESTD'
    SMA = 'SMA'
    MACD = 'MACD'
    RSI = 'RSI'
    BBANDS = 'BBANDS'
    Volatility = 'Volatility'
    LogVolatility = 'LogVolatility'


def validateFeatureInput(feature: dict) -> bool:
    if 'type' not in feature:
        return False

    if feature['type'] not in [e.value for e in DeriveFeature]:
        return False

    return True


def deriveFeature(featureListInput: [dict], data: dict[str, pd.DataFrame]) -> [str]:
    featureCols = set()

    for f in featureListInput:
        if not validateFeatureInput(f):
            continue

        for symbol, df in data.items():
            featureType = f['type']

            if featureType == DeriveFeature.EMA.value:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = EMA(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.STD.value:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = STD(df['close_return'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.ESTD.value:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = ESTD(df['close_return'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.SMA.value:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = SMA(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.RSI.value:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = RSI(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.MACD.value:
                fastPeriod = f['fastPeriod']
                slowPeriod = f['slowPeriod']
                signalPeriod = f['signalPeriod']

                macDColName = "macd_" + str(fastPeriod) + "_" + str(slowPeriod) + "_" + str(signalPeriod)
                macDSignalColName = "macdsignal_" + str(fastPeriod) + "_" + str(slowPeriod) + "_" + str(signalPeriod)
                macDHistColName = "macdhist_" + str(fastPeriod) + "_" + str(slowPeriod) + "_" + str(signalPeriod)

                macd, macdsignal, macdhist = MACD(df['close'], fastPeriod, slowPeriod, signalPeriod)
                df[macDHistColName] = macdhist
                df[macDColName] = macd
                df[macDSignalColName] = macdsignal

                featureCols.update([macDColName, macDSignalColName, macDHistColName])

            elif featureType == DeriveFeature.BBANDS.value:
                featureWindow = f['window']
                nbdevup = f['nbdevup']
                nbdevdn = f['nbdevdn']

                upper, middle, lower = BBANDS(df['close'], featureWindow, nbdevup, nbdevdn)
                upperFeatureColName = "bbands_upper_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)
                middleFeatureColName = "bbands_middle_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)
                lowerFeatureColName = "bbands_lower_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)

                df[upperFeatureColName] = upper
                df[middleFeatureColName] = middle
                df[lowerFeatureColName] = lower

                featureCols.update([upperFeatureColName, middleFeatureColName, lowerFeatureColName])

            elif featureType == DeriveFeature.Volatility.value:
                colName = featureType
                df[colName] = Volatility(df['close_return'])
                featureCols.add(colName)

            elif featureType == DeriveFeature.LogVolatility.value:
                colName = featureType
                df[colName] = np.log(Volatility(df['close_return']).iloc[1:])
                featureCols.add(colName)

    for df in data.values():
        df.dropna(inplace=True)

    return list(featureCols)


def basicFeature(featureListInput: [str], data: dict[str, pd.DataFrame]) -> [str]:
    featureCol = []

    for f in basicFeatures:
        if f not in featureListInput:
            for df in data.values():
                df.drop(columns=[f])
        elif f == 'close_return':
            for df in data.values():
                df['close_return'] = df['close'] / df['close'].shift(1) - 1

            featureCol.append(f)
        elif f == 'open_return':
            for df in data.values():
                df['open_return'] = df['open'] / df['open'].shift(1) - 1

            featureCol.append(f)
        elif f == 'high_return':
            for df in data.values():
                df['high_return'] = df['high'] / df['high'].shift(1) - 1

            featureCol.append(f)
        elif f == 'low_return':
            for df in data.values():
                df['low_return'] = df['low'] / df['low'].shift(1) - 1

            featureCol.append(f)
        elif f == 'log_volume':
            for df in data.values():
                df['log_volume'] = np.log(df['volume'])

            featureCol.append(f)
        elif f == 'log_high_low':
            for df in data.values():
                df['log_high_low'] = np.log(df['high'] - df['low'])

            featureCol.append(f)
        else:
            featureCol.append(f)

    for df in data.values():
        df.dropna(inplace=True)

    return featureCol


def transform(data: Dict[str, pd.DataFrame]) -> Dict[str, tuple]:
    table = dict()
    for symbol, df in data.items():
        y = np.array(df['close_return'].iloc[1:]).reshape(-1, 1)
        X = df.iloc[:-1, :]
        scalerFeatures = MinMaxScaler(feature_range=(-1, 1))
        scalerLabel = MinMaxScaler(feature_range=(-1, 1))
        scalerFeatures.fit(X)
        scalerLabel.fit(y)
        transform_features = scalerFeatures.transform(X)
        transform_label = scalerLabel.transform(y)
        table[symbol] = (transform_features, transform_label, scalerFeatures, scalerLabel)
    return table
