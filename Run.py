from AlphaEvolve import AlphaEvolve
from GetData import getData

if __name__ == '__main__':
    getData(fromDate = (2016, 1, 1))
    x = AlphaEvolve()
    x.run()