from AlphaEvolve import AlphaEvolve

if __name__ == '__main__':
    ## frequency : '5min', '15min', '1H', '4H', '1D', '1W', '2W', '1M'
    x = AlphaEvolve(population = 50, tournament = 10, window = 30, numNewAlphaPerMutation = 1000, trainRatio = 0.6, validRatio = 0.2, maxNumNodes = 200, maxLenShapeNode = 50)
    x.run()