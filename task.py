import pandas as pd
from AlphaEvolve import AlphaEvolve


def startAlphaEvolve(params: dict):
    for symbol in params['data'].keys():
        params['data'][symbol] = pd.read_json(params['data'][symbol], orient="index")

    x = AlphaEvolve(**params)
    x.run()
