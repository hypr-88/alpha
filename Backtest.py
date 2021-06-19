import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest(returns: np.ndarray, Prediction: np.ndarray, show: bool = True, pctLongShort: float = 0.1, rf: float = 0.02):
    '''
    This function takes actual returns and predicts returns matrix as parameters and run backtest with the strategy:
        + Equal-weighted buy top (pctLongShort*100)% symbols which have highest predict returns each day.
        + Equal-weighted sell top (pctLongShort*100)% symbols which have lowest predict returns each day.
    The total amount of long is 1 and total amount of short is 1.
    
    Parameters
    ----------
    returns : np.ndarray
        This parameter is a matrix of actual returns calculated from data. Each row represents one day and each column represent one symbol.
        It is calculated from value of attribute OperandsValues 's0' of class AlphaEvolve.
    Prediction : np.ndarray
        This parameter is a matrix of predicted returns calculated from Predict Operations. Each row represents one day and each column represent one symbol.
        It is calculated from value of attribute OperandsValues 's1' of class AlphaEvolve.
    show : bool, optional
        If show is True, it prints out the cumulative returns, anualized returns, anualied standard deviation, and sharpe ratio during the test period.
        It also plots a chart of daily returns. The default is True.
    pctLongShort : float, optional
        The percentage of number of longing/shorting symbols over total numbers of symbols. The default is 0.1.
    rf : float, optional
        Risk-free rate used to calculate sharpe ratio. The default is 0.02.

    Returns
    -------
    Tuples: (dailyReturns, annualizedReturns, sharpe)
    dailyReturns : np.ndarray
        Array of daily returns during the test period.
    annualizedReturns : float
        Anualized returns.
    sharpe : float
        sharpe ratio.

    '''
    
    # make sure the shape of actual and predict returns are the same.
    if returns.shape == Prediction.shape:
        # the total numbers of symbols
        noSymbol = returns.shape[1]
        
        #convert actual and predict returns to DataFrame
        returns = pd.DataFrame(returns)
        Prediction = pd.DataFrame(Prediction)
        
        #number of long symbols and short symbols
        noLongShort = int(np.ceil(noSymbol * pctLongShort))
        
        #weigths of each long/short symbol
        weights = 1/noLongShort
        
        #convert predict returns to ranking array
        Prediction = Prediction.rank(pct = True, axis = 1)
        
        #define assign weights function
        def assignWeight(row):
            #index of long/short symbols
            long = row.nlargest(noLongShort, keep = 'first').index.tolist()
            short = row.nsmallest(noLongShort, keep = 'last').index.tolist()
            #sometimes every entry in row is the same. Then appling .rank() method will cause 0s row. 
            #-> Then, we keep 'first' and 'last' to avoid overlaping index.
            
            #index of the others
            notTrade = [i for i in row.index if i not in long+short]
            
            #assign 0 weights to non-trade asset
            row[notTrade] = 0
            #assign positive weights to long asset
            row.loc[long] = weights
            #assign negative weights to short asset
            row.loc[short] = -weights
            return row
        
        #apply assignWeight to each row of prediction
        Weights = Prediction.apply(assignWeight, axis = 1)
        
        #calculate daily returns of portfolio
        dailyReturns = (Weights * returns).sum(axis = 1)
        
        #calculate cumulative returns and standard deviation
        cumulativeReturns = sum(dailyReturns)
        std = dailyReturns.std()
        
        #anualized returns and standard deviation
        annualizedReturns = cumulativeReturns/len(returns.index)*365
        annualizedSTD = std*np.sqrt(365/len(returns.index))
        
        #sharpe ratio
        sharpe = (annualizedReturns - rf)/annualizedSTD
        
        if show: 
            print('++++++++++++++++++++++++++++++++++++++++')
            print('CUMULATUVE RETURNS:', cumulativeReturns)
            print('ANNUAlIZED RETURNS:', annualizedReturns)
            print('ANNUALIZED STD    :', annualizedSTD)
            print('SHARPE RATIO      :', sharpe)
        
        #cumulative returns
        for i in range(1, len(dailyReturns)):
            dailyReturns[i] += dailyReturns[i-1]
        
        if show:
            plt.figure(figsize = (5,3))
            plt.plot(dailyReturns)
            plt.title('Cumulative Returns')
            plt.ylabel('Returns')
            plt.show()
            print('++++++++++++++++++++++++++++++++++++++++')
        
        return dailyReturns, annualizedReturns, sharpe