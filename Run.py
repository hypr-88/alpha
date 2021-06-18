from AlphaEvolve import AlphaEvolve
from GetData import getData

if __name__ == '__main__':
    ## frequency : '5min', '15min', '1H', '4H', '1D', '1W', '2W', '1M'
    getData('1D', file = "may_premium_dataset_BTC.zip")
    x = AlphaEvolve()
    x.run()