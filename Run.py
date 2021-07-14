from AlphaEvolve import AlphaEvolve

if __name__ == '__main__':
    ## frequency : '5min', '15min', '1H', '4H', '1D', '1W', '2W', '1M'
    x = AlphaEvolve(name = 'abc', population = 50, tournament = 10, window = 30, numNewAlphaPerMutation = 50, trainRatio = 0.42, validRatio = 0.57, maxNumNodes = 1000, maxLenShapeNode = 50)
    x.run()