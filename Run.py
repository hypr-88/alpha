from AlphaEvolve import AlphaEvolve

if __name__ == '__main__':
    ## frequency : '5min', '15min', '1H', '4H', '1D', '1W', '2W', '1M'
    x = AlphaEvolve(population = 100, tournament = 10, window = 30, numNewAlphaPerMutation = 10, trainRatio = 0.62, validRatio = 0.19, maxNumNodes = 500, maxLenShapeNode = 50)
    x.run()