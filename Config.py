import pandas as pd
from enum import Enum
from Features import EMA, ESTD, STD, SMA, RSI, MACD, BBANDS

basicFeatures = ['open', 'close', 'high', 'volume', 'return']


class DeriveFeature(Enum):
    EMA = 'EMA'
    STD = 'STD'
    ESTD = 'ESTD'
    SMA = 'SMA'
    MACD = 'MACD'
    RSI = 'RSI'
    BBANDS = 'BBANDS'


def validateFeatureInput(feature: dict) -> bool:
    if 'type' not in feature or 'window' not in feature:
        return False

    if feature['type'] not in DeriveFeature:
        return False

    return True


def deriveFeature(featureListInput: [dict], data: dict[str, pd.DataFrame]) -> [str]:
    featureCols = set()

    for f in featureListInput:
        if not validateFeatureInput(f):
            continue

        for symbol, df in data.items():
            featureType = f['type']

            if featureType == DeriveFeature.EMA:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = EMA(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.STD:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = STD(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.ESTD:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = ESTD(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.SMA:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = SMA(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.RSI:
                featureWindow = f['window']
                colName = featureType + "_" + str(featureWindow)
                df[colName] = RSI(df['close'], featureWindow)
                featureCols.add(colName)

            elif featureType == DeriveFeature.MACD:
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

                featureCols.update([macDHistColName, macDSignalColName, macDHistColName])

            elif featureType == DeriveFeature.BBANDS:
                featureWindow = f['window']
                nbdevup = f['nbdevup']
                nbdevdn = f['nbdevdn']

                upper, middle, lower = BBANDS(df['close'], featureWindow, nbdevup, nbdevdn)
                upperFeatureColName = "upper_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)
                middleFeatureColName = "middle_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)
                lowerFeatureColName = "lower_" + str(featureWindow) + "_" + str(nbdevup) + "_" + str(nbdevdn)

                df[upperFeatureColName] = upper
                df[middleFeatureColName] = middle
                df[lowerFeatureColName] = lower

                featureCols.update([upperFeatureColName, middleFeatureColName, lowerFeatureColName])

    return list(featureCols)


def basicFeature(featureListInput: [str], data: dict[str, pd.DataFrame]):
    for f in basicFeatures:
        if f not in featureListInput:
            for df in data.values():
                df.drop(f)
        elif f == 'return':
            for df in data.values():
                df['return'] = df['close'] / df['close'].shift(1) - 1
