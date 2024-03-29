import multiprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import rankdata
from sklearn.preprocessing import MinMaxScaler

import copy
from Operands import Scalar, Vector, Matrix
from Graph import Graph
from Alpha import Alpha
from OPs import *
from Features import *
from Backtest import backtest

from datetime import datetime
from dateutil.relativedelta import relativedelta

from zipfile import ZipFile
import os
import requests
import json
np.seterr(all="ignore") 
pd.options.mode.chained_assignment = None
class AlphaEvolve():
    '''
    Main class to run alpha evolve
        ...
    
    Attributes
    ----------
    
    startTime : datetime.datetime
        The time we start evolving
        
    endTime : datetime.datetime
        The time we stop evolving
    
    populationLength : int
        The length of population
    
    tournamentLength : int
        The length of tournament. Must be less than populationLength
    
    window : int
        The window length when we fit the inputs
    
    numNewAlphaPerMutation : int
        This is an expansion from AlphaEvolve. 
        In the paper, the algorithm is mutating best fit alpha in the tournament and add it to the population and delete the oldest alpha.
        In this algo, we mutate best fit alpha in the tournament to create numNewAlphaPerMutation new alphas. We then compare the fitness and add the best fit of these new alpha and add it to population.
    
    trainRatio : float
        The ratio of train data over all collected data.
    
    trainLength : int
        The length of train data
        
    validRatio : float
        The ratio of validation data over all collected data.
    
    validLength : int
        The length of validation data
    
    population : list
        The list of object Alpha defined in Alpha.py
    
    fitnessScore : dict
        The dictionary whose keys are fingerprint of each alpha and values are fitness score of the respective alpha.
        
    currAlpha : Alpha
        The current alpha object that this class point to.
    
    symbolList : list
        List of all symbols
    
    data : dict
        dictionary used to store the data after processing
        keys : str : symbol
        values : tuple : (Industry:str, Data: pd.DataFrame)
    
    dataLength : int
        The length of data
        
    featuresList : list
        The list of features we used
    
    kAlphas : dict
        Each symbol need a different alpha object to execute setup/predict/update.
        -> This dictionary store replication of currAlpha for each symbol.
        
        keys : str : symbol
        values : Alpha
        
        ***NOTE*** attribute currAlpha always points to kAlphas[symbolList[0]]
    
    OperandsValues : dict
        This dictionary store value of each nodes for all symbols
        keys : str : node.
        values : list : list of value of each node.
        
        Ex: {'s1': [0.1, -0.2, 0.02], 's0': [-0.1, 0., 0.], 'v1': [np.array([1,1,1]), np.array([1,1,1]), np.array([0,0,0])]}
    
    bestFit : list [Alpha, fitnessScore, returnsArray, annualizedReturns, sharpe]
        The list of information of the best fit alpha ever (from strating time)
        
    Methods
    -------
    
    __init__(graph: Graph = None, population: int = 5, tournament: int = 2, window: int = 20, numNewAlphaPerMutation: int = 1, trainRatio: float = 0.8, validRatio: float = 0.1, TimeBudget: tuple = (1, 0, 0, 0))
        initiate AlphaEvolve
        
    checkTimeBudget()
        check whether continue evolving or not
    
    initiateAlpha(graph: Graph = None)
        Create the first Alpha
    
    importData()
        import data from .csv files in 'RawData/'
    
    preparedData()
        process data includes: add features, calculate returns, normalized and store them in attribute data
    
    createWindow(symbol: str, i: int):
        get the window input and actual return output of symbol on date i-th
    
    initiatePopulation()
        Create the first population by mutating the first initiated alpha.
        
    runFirstPopulation()
        Evaluate all alpha in the first population
        
    evaluate()
        evaluate the currAlpha and return fitness score, prediction matrix and actual matrix returns during validation period and test period.
        
    replicateAlpha()
        Reset kAlpha attribute and replicate currAlpha and store them to kAlpha dictionary.
        
    resetOperandsValues()
        After evaluate 1 alpha, we need to reset the OperandsValues dictionary to evaluate the new currAlpha
    
    runFirstWindow()
        fit the first window input to alpha.
        The purpose is to make sure every vector and matrix nodes have shape (!= None) and the operations are appropriate.
        While running this method, we also run executionOperation method which can handle the None value input (new Scalar/Vector/Matrix) 
        and delete all those operations which are not valid.
        
        Ex: after mutating, we have these operations:
            1. ['v3', 20, ['v1']] ~ v3 = 1/v1
            2. ['v2', 22, ['v2']] ~ v2 = |v3|
            while shape of v1 is 10, shape of v2 is 5 and shape of v3 is None.
            after 1st operation, shape of v3 is 10 -> 2nd operation is invalid -> delete 2nd operation
            -> method runFirstWindow() handle the shape of vector and matrix nodes.
    
    train()
        train the current alpha using train data
    
    validate()
        evaluate and predict the current alpha using validation data
    
    test()
        evaluate and predict the current alpha using the test data
        
    setup()
        execute all operation in setup step for all Alphas of each symbol
    
    addM0(i: int)
        add window input at the date i-th to operand m0
    
    predict()
        execute all operation in predict step for all Alphas of each symbol
    
    addS0(i: int)
        add output returns at the date i-th to operand s0
    
    update()
        execute all operation in update step for all Alphas of each symbol
    
    prunning()
        prunning the current Alpha
    
    fingerprint()
        create fingerprint of the current Alpha
    
    summaryAlpha()
        print out basic metric after evaluating the current Alpha
    
    summaryBestFit()
        print out basic metric of the best fitness Alpha since the first time
    
    getBestFit()
        get the best fitness score of alpha in attribute tournament and point currAlpha to that alpha
    
    mutate()
        mutate the currAlpha
    
    pickTournament()
        randomly choose tournamentLength alphas in population and bestfit alpha
        -> this method facilitate to choose best fit alpha ever (even if that alpha is not in population)
    
    executeOperation(Operation: list)
        This function execute Operation by:
            1. handle None value/shape operands (input or output)
            2. check shape/value valid or not. If not valid -> delete the Operation in all alphas in kAlpha
            3. if valid, update value of output node of all alphas in kAlpha
                        and update value of output node of OperandsValues dictionary
    
    run()
        run the alpha evolve
    '''
    def __init__(self, name = 'abc', mutateProb = 0.9, graph: Graph = None, population: int = 25, tournament: int = 10, window: int = 20, numNewAlphaPerMutation: int = 3,
                 trainRatio: float = 0.3, validRatio: float = 0.1, testRatio: float = 0.1, TimeBudget: tuple = (1, 0, 0, 0), frequency: str = '1D', maxNumNodes: tuple = (50, 200, 250), maxLenShapeNode: int = 30,
                 file: str = "Binance_30m_spot_Full.zip", addProb: float = 0.4, delProb: float = 0.3, changeProb: float = 0.3, num_of_observ: int = 8000, pctLongShort: float = 0.3):
        '''
        Method used to initiate AlphaEvolve

        Parameters
        ----------
        graph : Graph, optional
            Initial Graph. We can create graph to represent NN or any initial Alpha. If None, algo starts from no operation. The default is None.
        population : int, optional
            the length of population. The default is 25.
        tournament : int, optional
            the length of tournament. The default is 10.
        window : int, optional
            the length of window input. The default is 20.
        numNewAlphaPerMutation : int, optional
            the number of new alphas mutated in each mutate step. The default is 1.
        trainRatio : float, optional
            the ratio of train data. The default is 0.8.
        validRatio : float, optional
            the ration of validation data. The default is 0.1.
        TimeBudget : tuple, optional
            the time to evolve. (days, hours, minutes, seconds). The default is (1, 0, 0, 0).
        
        Returns
        -------
        None.

        '''
        self.scaler = {}
        self.name = name
        self.mutateProb = mutateProb
        
        self.startTime = datetime.now()
        days, hours, minutes, seconds = TimeBudget
        self.endTime = self.startTime + relativedelta(days = days, hours = hours, minutes = minutes, seconds = seconds)
        
        self.frequency = frequency
        self.file = file
        self.num_of_observ = num_of_observ
        
        self.populationLength = population
        self.tournamentLength = tournament
        self.window = window
        self.numNewAlphaPerMutation = numNewAlphaPerMutation
        self.pctLongShort = pctLongShort
        
        self.trainRatio = trainRatio
        self.validRatio = validRatio
        self.testRatio = testRatio
        
        self.addProb = addProb
        self.delProb = delProb
        self.changeProb = changeProb
        
        self.population = []
        self.fitnessScore = {}
        
        self.maxNumNodes = maxNumNodes
        self.maxLenShapeNode = maxLenShapeNode
        self.initiateAlpha(graph)
        
        self.mutateSummary = []
        self.fitnessSummary = []
        self.sharpeSummary = []
        self.testScoreSummary = []
        self.allAlphaInformation = []
        
    def checkTimeBudget(self):
        '''
        check whether continue evolving or not

        Returns
        -------
        bool
            if true, continue evolving, else, stop evolving

        '''
        return datetime.now() < self.endTime
    
    def initiateAlpha(self, graph: Graph = None):
        '''
        Create the first Alpha

        Parameters
        ----------
        graph : Graph, optional
            Initial Graph. We can create graph to represent NN or any initial Alpha. The default is None.

        Returns
        -------
        None.

        '''
        self.currAlpha = Alpha(graph, maxNumNodes = self.maxNumNodes, mutateProb = self.mutateProb, rf = 0.0001, maxLenShapeNode = self.maxLenShapeNode, addProb = self.addProb, delProb = self.delProb, changeProb = self.changeProb)

    def parallelNewMutate(self, alpha: Alpha):
        self.currAlpha = alpha
        # prunning and get fingerprint
        self.prunning()
        fingerPrint = self.fingerprint()
        fitnessScore, dailyReturns, annualizedReturns, sharpe, testScore, ultilization, returnDetails, returns, Prediction = self.summaryAlpha()
        return alpha, fingerPrint, fitnessScore, dailyReturns, annualizedReturns, sharpe, self.OperandsValues, testScore, ultilization, returnDetails, returns, Prediction
    
    def combineAlphas(self, alpha_1, alpha_2):
        graph_1 = copy.deepcopy(alpha_1.graph)
        graph_2 = copy.deepcopy(alpha_2.graph)
        
        nodes = {}
        for key, value in graph_1.nodes.items():
            if key not in {'m0', 's1', 's0'}:
                nodes[key+'1'] = copy.deepcopy(value)
            elif key in {'m0', 's0'}:
                nodes[key] = copy.deepcopy(value)
            elif key == 's1':
                nodes[key+'1'] = copy.deepcopy(value)
                nodes[key] = copy.deepcopy(value)
        for key, value in graph_2.nodes.items():
            if key not in {'m0', 's0'}:
                nodes[key+'2'] = copy.deepcopy(value)
        nodes['s3'] = Scalar()
        nodes['s4'] = Scalar(2)
        
        setupOPs_1 = graph_1.setupOPs.copy()
        setupOPs_2 = graph_2.setupOPs.copy()
        predictOPs_1 = graph_1.predictOPs.copy()
        predictOPs_2 = graph_2.predictOPs.copy()
        updateOPs_1 = graph_1.updateOPs.copy()
        updateOPs_2 = graph_2.updateOPs.copy()
        
        newSetup = []
        newPredict = []
        newUpdate = []
        for operation in setupOPs_1:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '1'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '1'
            newSetup.append([out, op, inps])
        
        for operation in setupOPs_2:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '2'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '2'
            newSetup.append([out, op, inps])
        
        for operation in predictOPs_1:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '1'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '1'
            newPredict.append([out, op, inps])
        
        for operation in predictOPs_2:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '2'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '2'
            newPredict.append([out, op, inps])
        
        for operation in updateOPs_1:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '1'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '1'
            newUpdate.append([out, op, inps])
        
        for operation in updateOPs_2:
            out, op, inps = operation
            if isinstance(out, str) and out not in {'m0', 's0'}: out += '2'
            for i, inp in enumerate(inps):
                if isinstance(inp, str) and inp not in {'m0', 's0'}: inps[i] += '2'
            newUpdate.append([out, op, inps])
        
        newPredict.append(['s3', 1, ['s11', 's12']])
        newSetup.append(['s4', 56, [2]])
        newPredict.append(['s1', 4, ['s3', 's4']])
        
        maxNumNodes = (graph_1.maxNumNodes[0] + graph_2.maxNumNodes[0], graph_1.maxNumNodes[1] + graph_2.maxNumNodes[1], graph_1.maxNumNodes[2] + graph_2.maxNumNodes[2])
        
        newGraph = Graph(nodes = nodes, setupOPs = newSetup, predictOPs = newPredict, updateOPs = newUpdate, maxNumNodes = maxNumNodes)
        newAlpha = Alpha(newGraph)
        return newAlpha
        
    def createNewMutate(self):
        self.pickTournament()
        bestFit_1, alpha_1 = self.getBestFit(self.tournament)
        bestFit_2, alpha_2 = self.getBestFit([alpha for alpha in self.tournament if alpha != alpha_1])
        newMutate = []
        summary = {'setup': {'prob': {'add': [], 'del': [], 'change': []}, 
                          'actual': {'add': 0, 'del': 0, 'change': 0}}, 
                'predict': {'prob': {'add': [], 'del': [], 'change': []}, 
                            'actual': {'add': 0, 'del': 0, 'change': 0}}, 
                'update': {'prob': {'add': [], 'del': [], 'change': []}, 
                           'actual': {'add': 0, 'del': 0, 'change': 0}}}
        for i in range(self.numNewAlphaPerMutation):
            cnt = 0
            while True:
                try:
                    if bestFit_2 <= 0:
                        newAlpha = copy.deepcopy(alpha_1)
                    elif np.random.binomial(1, bestFit_1/(bestFit_1 + bestFit_2)):
                        newAlpha = self.combineAlphas(alpha_1, alpha_2)
                    else:
                        newAlpha = copy.deepcopy(alpha_1)
                    #newAlpha = copy.deepcopy(alpha_1)
                    mutateSummary = newAlpha.mutate()
                    cnt += 1
                    if not newAlpha.checkS1ConnectsM0_Predict() and cnt < 100:
                        continue
                    newMutate.append(newAlpha)
                    
                    summary['setup']['prob']['add'].append(mutateSummary['setup']['prob']['add'])
                    summary['setup']['prob']['del'].append(mutateSummary['setup']['prob']['del'])
                    summary['setup']['prob']['change'].append(mutateSummary['setup']['prob']['change'])
                    summary['setup']['actual']['add'] += mutateSummary['setup']['actual']['add']
                    summary['setup']['actual']['del'] += mutateSummary['setup']['actual']['del']
                    summary['setup']['actual']['change'] += mutateSummary['setup']['actual']['change']
                    
                    summary['predict']['prob']['add'].append(mutateSummary['predict']['prob']['add'])
                    summary['predict']['prob']['del'].append(mutateSummary['predict']['prob']['del'])
                    summary['predict']['prob']['change'].append(mutateSummary['predict']['prob']['change'])
                    summary['predict']['actual']['add'] += mutateSummary['predict']['actual']['add']
                    summary['predict']['actual']['del'] += mutateSummary['predict']['actual']['del']
                    summary['predict']['actual']['change'] += mutateSummary['predict']['actual']['change']
                    
                    summary['update']['prob']['add'].append(mutateSummary['update']['prob']['add'])
                    summary['update']['prob']['del'].append(mutateSummary['update']['prob']['del'])
                    summary['update']['prob']['change'].append(mutateSummary['update']['prob']['change'])
                    summary['update']['actual']['add'] += mutateSummary['update']['actual']['add']
                    summary['update']['actual']['del'] += mutateSummary['update']['actual']['del']
                    summary['update']['actual']['change'] += mutateSummary['update']['actual']['change']
                    break
                except:
                    continue
        
        summary['setup']['prob']['add'] = sum(summary['setup']['prob']['add'])/len(summary['setup']['prob']['add'])
        summary['setup']['prob']['del'] = sum(summary['setup']['prob']['del'])/len(summary['setup']['prob']['del'])
        summary['setup']['prob']['change'] = sum(summary['setup']['prob']['change'])/len(summary['setup']['prob']['change'])
        
        summary['predict']['prob']['add'] = sum(summary['predict']['prob']['add'])/len(summary['predict']['prob']['add'])
        summary['predict']['prob']['del'] = sum(summary['predict']['prob']['del'])/len(summary['predict']['prob']['del'])
        summary['predict']['prob']['change'] = sum(summary['predict']['prob']['change'])/len(summary['predict']['prob']['change'])
        
        summary['update']['prob']['add'] = sum(summary['update']['prob']['add'])/len(summary['update']['prob']['add'])
        summary['update']['prob']['del'] = sum(summary['update']['prob']['del'])/len(summary['update']['prob']['del'])
        summary['update']['prob']['change'] = sum(summary['update']['prob']['change'])/len(summary['update']['prob']['change'])
        
        self.mutateSummary.append(summary)
        return newMutate
    
    def save_and_plot(self):
        df = pd.DataFrame()
        df['setup_prob_add'] = [mutateSummary['setup']['prob']['add'] for mutateSummary in self.mutateSummary]
        df['setup_prob_del'] = [mutateSummary['setup']['prob']['del'] for mutateSummary in self.mutateSummary]
        df['setup_prob_change'] = [mutateSummary['setup']['prob']['change'] for mutateSummary in self.mutateSummary]
        df['setup_actual_add'] = [mutateSummary['setup']['actual']['add'] for mutateSummary in self.mutateSummary]
        df['setup_actual_del'] = [mutateSummary['setup']['actual']['del'] for mutateSummary in self.mutateSummary]
        df['setup_actual_change'] = [mutateSummary['setup']['actual']['change'] for mutateSummary in self.mutateSummary]
        
        df['predict_prob_add'] = [mutateSummary['predict']['prob']['add'] for mutateSummary in self.mutateSummary]
        df['predict_prob_del'] = [mutateSummary['predict']['prob']['del'] for mutateSummary in self.mutateSummary]
        df['predict_prob_change'] = [mutateSummary['predict']['prob']['change'] for mutateSummary in self.mutateSummary]
        df['predict_actual_add'] = [mutateSummary['predict']['actual']['add'] for mutateSummary in self.mutateSummary]
        df['predict_actual_del'] = [mutateSummary['predict']['actual']['del'] for mutateSummary in self.mutateSummary]
        df['predict_actual_change'] = [mutateSummary['predict']['actual']['change'] for mutateSummary in self.mutateSummary]
        
        df['update_prob_add'] = [mutateSummary['update']['prob']['add'] for mutateSummary in self.mutateSummary]
        df['update_prob_del'] = [mutateSummary['update']['prob']['del'] for mutateSummary in self.mutateSummary]
        df['update_prob_change'] = [mutateSummary['update']['prob']['change'] for mutateSummary in self.mutateSummary]
        df['update_actual_add'] = [mutateSummary['update']['actual']['add'] for mutateSummary in self.mutateSummary]
        df['update_actual_del'] = [mutateSummary['update']['actual']['del'] for mutateSummary in self.mutateSummary]
        df['update_actual_change'] = [mutateSummary['update']['actual']['change'] for mutateSummary in self.mutateSummary]
        
        df.to_csv('mutateSummary.csv')
        
        plt.figure(figsize = (10,8))
        plt.plot(df['setup_prob_add'], label = 'mutation by adding')
        plt.plot(df['setup_prob_del'], label = 'mutation by removing')
        plt.plot(df['setup_prob_change'], label = 'mutation by changing')
        plt.title('Setup Probability')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize = (10,8))
        plt.plot(df['setup_actual_add'], label = 'mutation by adding')
        plt.plot(df['setup_actual_del'], label = 'mutation by removing')
        plt.plot(df['setup_actual_change'], label = 'mutation by changing')
        plt.title('Setup Actual')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize = (10,8))
        plt.plot(df['predict_prob_add'], label = 'mutation by adding')
        plt.plot(df['predict_prob_del'], label = 'mutation by removing')
        plt.plot(df['predict_prob_change'], label = 'mutation by changing')
        plt.title('Predict Probability')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize = (10,8))
        plt.plot(df['predict_actual_add'], label = 'mutation by adding')
        plt.plot(df['predict_actual_del'], label = 'mutation by removing')
        plt.plot(df['predict_actual_change'], label = 'mutation by changing')
        plt.title('Predict Actual')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize = (10,8))
        plt.plot(df['update_prob_add'], label = 'mutation by adding')
        plt.plot(df['update_prob_del'], label = 'mutation by removing')
        plt.plot(df['update_prob_change'], label = 'mutation by changing')
        plt.title('Update Probability')
        plt.legend()
        plt.show()
        
        
        plt.figure(figsize = (10,8))
        plt.plot(df['update_actual_add'], label = 'mutation by adding')
        plt.plot(df['update_actual_del'], label = 'mutation by removing')
        plt.plot(df['update_actual_change'], label = 'mutation by changing')
        plt.title('Update Actual')
        plt.legend()
        plt.show()
        
        
        
        df1 = pd.DataFrame()
        df1['maxFit'] = [x[0] for x in self.fitnessSummary]
        df1['aveFit'] = [x[1] for x in self.fitnessSummary]
        df1['minFit'] = [x[2] for x in self.fitnessSummary]
        
        df1['maxSharpe'] = [x[0] for x in self.sharpeSummary]
        df1['aveSharpe'] = [x[1] for x in self.sharpeSummary]
        df1['minSharpe'] = [x[2] for x in self.sharpeSummary]
        
        df1['TestScore'] = self.testScoreSummary
        
        df1.to_csv('evolve summary.csv')
        
        plt.figure(figsize = (10,6))
        plt.plot(df1['maxFit'], label = 'max')
        plt.plot(df1['aveFit'], label = 'ave')
        plt.plot(df1['minFit'], label = 'min')
        plt.title('Fitness Score')
        plt.legend()
        plt.show()
        
        plt.figure(figsize = (10,6))
        plt.plot(df1['maxFit'], label = 'validation')
        plt.plot(df1['TestScore'], label = 'test')
        plt.legend()
        plt.show()
        
        plt.figure(figsize = (10,6))
        plt.plot(df1['maxSharpe'], label = 'max')
        plt.plot(df1['aveSharpe'], label = 'ave')
        plt.plot(df1['minSharpe'], label = 'min')
        plt.title('Sharpes')
        plt.legend()
        plt.show()
    
    def run(self):
        '''
        Evolving Method

        Returns
        -------
        None.

        '''
        #import data
        self.importData()
        self.preparedData()
        self.trainLength = round(self.dataLength * self.trainRatio)
        self.validLength = round(self.dataLength * self.validRatio)
        self.testLength = round(self.dataLength * self.testRatio)
        
        #initiate 1st population
        #self.initiatePopulation()
        self.runFirstPopulation()
        self.summaryBestFit()
        self.cont()
        self.last_check()
        
    def cont(self):
        cnt = 0
        while self.checkTimeBudget():
            newMutate = self.createNewMutate()
            
            validAlpha = []
            try:
                ctx = multiprocessing.get_context('spawn')
                pool = ctx.Pool()
                count = 0
                for alpha, fingerPrint, fitnessScore, dailyReturns, annualizedReturns, sharpe, OperandsValues, testScore, ultilization, returnDetails, returns, Prediction in pool.imap(self.parallelNewMutate, newMutate):
                    if fingerPrint in self.fitnessScore:
                        pass
                    elif (fitnessScore > -1) and alpha.checkS1ConnectsM0_Predict():# and np.corrcoef(dailyReturns, self.bestFit[2])[0,1] < 0.9:  # fitnessScore = -1 implies s1 does not connect to m0 (we set this value in method evaluate()) -> do not add to population
                        validAlpha.append(
                            [alpha, fitnessScore, dailyReturns, annualizedReturns, sharpe, OperandsValues, testScore, ultilization, returnDetails, returns, Prediction])
                        self.allAlphaInformation.append([alpha, fitnessScore, dailyReturns, annualizedReturns, sharpe, OperandsValues, testScore, ultilization, returnDetails, returns, Prediction])
                    
                    print("Done:", count)
                    count += 1
    
                pool.close()
                pool.join()
            except ValueError:
                #ValueError: Input contains infinity or a value too large for dtype('float64')
                #avoid error when inverse-scaling
                continue
            
            if validAlpha == []:
                print('continue')
                continue
            maxFit = max([x[1] for x in validAlpha])
            minFit = min([x[1] for x in validAlpha])
            aveFit = sum([x[1] for x in validAlpha])/len([x[1] for x in validAlpha])
            maxSharpe = max([x[4] for x in validAlpha])
            minSharpe = min([x[4] for x in validAlpha])
            aveSharpe = sum([x[4] for x in validAlpha])/len([x[4] for x in validAlpha])
            self.fitnessSummary.append((maxFit, aveFit, minFit))
            self.sharpeSummary.append((maxSharpe, aveSharpe, minSharpe))
            #sort new mutated alphas by decreasing order of fitness
            newAlphaInfo = sorted(validAlpha, key = lambda x: x[1], reverse = True)[0]
            TestScore = newAlphaInfo[6]
            self.testScoreSummary.append(TestScore)
            newAlphaInfo[0].graph.show()
            print('++++++++++++++++++++++++++++++++++++++++')
            print('VALIDATION FITNESS:', newAlphaInfo[1])
            print('ANNUAlIZED RETURNS:', newAlphaInfo[3])
            print('SHARPE RATIO      :', newAlphaInfo[4])
            newAlphaInfo[7].index = self.symbolList
            print('ULTILIZATION      :')
            print(newAlphaInfo[7])
            print('Returns details   :')
            newAlphaInfo[8].index = self.symbolList
            print(newAlphaInfo[8])
            plt.figure(figsize = (5,3))
            plt.plot(newAlphaInfo[2])
            plt.title('Cumulative Returns')
            plt.ylabel('Returns')
            plt.show()
            print('++++++++++++++++++++++++++++++++++++++++')
            
            #get best fit alpha in new mutated alphas
            self.currAlpha = newAlphaInfo[0]
            fingerPrint = self.fingerprint()
            self.fitnessScore[fingerPrint] = newAlphaInfo[1]
            
            # if this new alpha beat best fit alpha ever -> replace best fit alpha
            if newAlphaInfo[1] > self.bestFit[1]:
                self.bestFit = newAlphaInfo
                pd.DataFrame(newAlphaInfo[9]).to_csv('Actual returns.csv')
                pd.DataFrame(newAlphaInfo[10]).to_csv('Prediction.csv')
            
            # add alpha to population and remove oldest alpha
            self.population.append(self.currAlpha)
            self.population.pop(0)
            
            #summary best fit alpha ever
            #self.summaryBestFit()
            
            cnt+=1
            if cnt%10 == 0:
                self.save_and_plot()
        self.extractAlpha(self.name)
        
    def importData(self):
        '''
        import data from .csv files in 'RawData/'

        Returns
        -------
        None.

        '''
        main_curr = ['ADA', 'BCH', 'BNB', 'BTC', 'DASH', 'EOS', 'ETH', 'LTC', 'NEO', 'TRX', 'XEM', 'XLM', 'XMR', 'XRP', 'ZEC', 'USDS']
        main_pairs = [pair+'USDT' for pair in main_curr]
        self.data = {}
        with ZipFile(self.file, "r") as zip_ref:
            # Get list of files names in zip
            list_of_files = zip_ref.namelist()
           
            # Iterate over the list of file names in given list
            for elem in list_of_files:
                #get the symbol
                file = os.path.splitext(elem)[0]
                symbol = file.split('-')[0]
                if symbol not in main_pairs:
                    continue
            #for symbol in main_pairs:
                #read csv
                with zip_ref.open(file+'.csv') as f:
                    df = pd.read_csv(f)
                    
                    #drop columns
                    df.drop(columns = ['stock_code', 'money', 'factor', 'change', 'TradeCount', 
                                       'TakerBuyBaseVolume', 'TakerBuyQuoteVolume'], inplace = True)
                    
                    #rename the rest columns
                    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
                    #change time: str -> datetime
                    df['time'] = pd.to_datetime(df['time'])
                    
                    #grouped data to with frequency
                    new_df = pd.DataFrame()
                    new_df['open'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['open'].first()
                    new_df['high'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['high'].max()
                    new_df['low'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['low'].min()
                    new_df['close'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['close'].last()
                    new_df['volume'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['volume'].sum()
                    
                    df['vol x close'] = df['close']*df['volume']
                    new_df['VWAP'] = df.groupby(pd.Grouper(key = 'time', freq = self.frequency))['vol x close'].sum()/new_df['volume']
            
                    #drop nan
                    new_df.dropna(inplace = True)
                    num_of_observ = min(self.num_of_observ, len(new_df))
                    #filter data have start date before 2020 and end date on 2021-5-31
                    if new_df.index[-1] >= datetime(2021, 9, 9) and new_df.index[0] <= datetime(2021, 1, 1):
                        self.data[symbol] = new_df.iloc[-num_of_observ:]
        
        self.symbolList = list(self.data.keys())
        #get intersection trading date of all symbols
        idx = list(self.data.values())[0].index
        for symbol, df in self.data.items():
            idx = idx.intersection(df.index)
        
        for symbol, df in self.data.items():
            self.data[symbol] = df.loc[idx]
            print(symbol)
        
    def preparedData(self):
        '''
        add features, calculate returns, normalized and store them in attribute data

        Returns
        -------
        None.

        '''
        self.featuresList = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 'open_return', 'high_return', 'low_return', 'close_return', 
                             'log volume', 'log volatility', 'open-close', 'log high-low', 'EMA5', 'EMA10', 'EMA20', 'EMA30', 'STD5', 'STD10', 'STD20', 'STD30', 
                             'BB high', 'BB mid', 'BB low', 'MACD fast', 'MACD slow', 'MACD']
            
        for symbol in self.symbolList:
            # features
            df = self.data[symbol]
            df['open_return'] = df['open']/df['open'].shift(1) - 1
            df['high_return'] = df['high']/df['high'].shift(1) - 1
            df['low_return'] = df['low']/df['low'].shift(1) - 1
            df['close_return'] = df['close']/df['close'].shift(1) - 1
            df['log volume'] = np.log(df['volume'])
            df['log volatility'] = Volatility(df['close_return'].iloc[1:])
            df['log high-low'] = np.log(df['high'] - df['low'])
            df['open-close'] = (df['open'] - df['close'])
            df['EMA5'] = EMA(df['close'], 5)
            df['EMA10'] = EMA(df['close'], 10)
            df['EMA20'] = EMA(df['close'], 20)
            df['EMA30'] = EMA(df['close'], 30)
            df['STD5'] = ESTD(df['close_return'], 5)
            df['STD10'] = ESTD(df['close_return'], 10)
            df['STD20'] = ESTD(df['close_return'], 20)
            df['STD30'] = ESTD(df['close_return'], 30)
            df['BB high'], df['BB mid'], df['BB low'] = BBANDS(df['close'], 20, 2, 2)
            df['MACD fast'], df['MACD slow'], df['MACD'] = MACD(df['close'], 12, 26, 9)
            
            #normalize
            #for col in self.featuresList:
            #    df[col] /= df[col].max(skipna = True)
            #remove nan
            df.dropna(inplace = True)
            self.dataLength = len(df)
    def transformData(self):
        
        self.transformedData = {}
        for symbol in self.symbolList:
            X = self.data[symbol].iloc[:-1,:]
            y = np.array(self.data[symbol]['close_return'].iloc[1:]).reshape(-1, 1)
            scaler_features = MinMaxScaler(feature_range=(-1, 1))
            scaler_label = MinMaxScaler(feature_range=(-1, 1))
            scaler_features.fit(X)
            scaler_label.fit(y)
            transform_features = scaler_features.transform(X)
            transform_label = scaler_label.transform(y)
            self.transformedData[symbol] = transform_features, transform_label
            self.scaler.append((scaler_features, scaler_label))
        
    def createWindow(self, symbol: str, i: int):
        '''
        get the window input and actual return output of symbol on date i-th

        Parameters
        ----------
        symbol : str
        i : int
            index i-th represent the date in the processed data.

        Returns
        -------
        tuple
            X, y

        ''''''
        if i<self.window:
            return None
        else:
            X = self.data[symbol].iloc[:i,:]
            y = np.array(self.data[symbol]['close_return'].iloc[1:i+1]).reshape(-1, 1)
            scaler_features = MinMaxScaler(feature_range=(-1, 1))
            scaler_label = MinMaxScaler(feature_range=(-1, 1))
            scaler_features.fit(X)
            scaler_label.fit(y)
            transform_features = scaler_features.transform(X)
            transform_label = scaler_label.transform(y)
            self.scaler[symbol] = (scaler_features, scaler_label)
            
            X = transform_features[-self.window:,:]
            y = transform_label[-1,0]
            assert X.shape == (self.window, len(self.featuresList))
            assert y.shape == ()
            return np.array(X, dtype = np.float64), np.array(y, dtype = np.float64)'''
        
        
        if i<self.window:
            return None
        else:
            X = self.data[symbol].iloc[:i,:]
            #normalize
            for col in self.featuresList:
                X.loc[:,col] /= X.loc[:,col].max(skipna = True)
            
            X = X.iloc[-self.window:]
            y = self.data[symbol]['close_return'].iloc[i]
            return np.array(X, dtype = np.float64), np.array(y, dtype = np.float64)

    def initiatePopulation(self):
        '''
        Create the first population by mutating the first initiated alpha.

        Returns
        -------
        None.

        '''
        for i in range(self.populationLength - len(self.population.copy())):
            '''
            while True:
                try:
                    newAlpha = copy.deepcopy(self.currAlpha)
                    newAlpha.mutate()
                    self.population.append(newAlpha)
                    break
                except:
                    continue'''
            newAlpha = copy.deepcopy(self.currAlpha)
            newAlpha.mutate()
            self.population.append(newAlpha)

    def parallelPopulation(self, alpha: Alpha):
        print("Start alpha...")
        self.currAlpha = alpha
        # prunning
        self.prunning()
        # evaluation Alpha
        fitnessScore, dailyReturns, annualizedReturns, sharpe, testScore, ultilization, returnDetails, returns, Prediction = self.summaryAlpha()
        # fingerprint
        fingerPrint = self.fingerprint()

        return alpha, fingerPrint, fitnessScore, dailyReturns, annualizedReturns, sharpe, self.OperandsValues

    def runFirstPopulation(self):
        '''
        Evaluate all alpha in the first population
        Returns
        -------
        None.
        '''
        while True:
            count = len(self.population)
            self.initiatePopulation()
            try:
                self.bestFit = [None, -1000, 0, 0, 0, {}]
                ctx = multiprocessing.get_context('spawn')
                pool = ctx.Pool()
                for alpha, fingerPrint, fitnessScore, dailyReturns, annualizedReturns, sharpe, OperandsValues in \
                    pool.imap(self.parallelPopulation, self.population[:len(self.population)]):
        
                    if alpha.fingerprint() in self.fitnessScore:
                        pass
                    else:
                        self.fitnessScore[alpha.fingerprint()] = fitnessScore
        
                    # update best fit alpha ever
                    if fitnessScore > self.bestFit[1]:
                        self.bestFit = [alpha, fitnessScore, dailyReturns, annualizedReturns, sharpe, OperandsValues]
                    
                    self.population[count] = alpha
                    print("Done: ", count)
                    count += 1
                pool.close()
                pool.join()
                break
            except ValueError:
                self.population = self.population[:count]
                continue
    
    def evaluate(self):
        '''
        evaluate the currAlpha and return fitness score, prediction matrix and actual matrix returns during validation period and test period.

        Returns
        -------
        fitnessScore : float
            fitness score calculated from validation data.
        validPrediction : np.ndarray
            prediction 's1' calculated from validation data.
        validActual : np.ndarray
            actual returns 's0' from validation data.
        testScore : float
            fitness score calculated from test data.
        testPrediction : np.ndarray
            prediction 's1' calculated from test data.
        testActual : np.ndarray
            actual returns 's0' from test data. 

        '''
        #predefined shapes of operands
        self.runFirstWindow()
        
        #start evaluate
        self.replicateAlpha()
        self.resetOperandsValues()
        #train
        self.train()
        #validate
        fitnessScore, validPrediction, validActual = self.validate()
        #test
        testScore, testPrediction, testActual = self.test()
        #result
        return fitnessScore, validPrediction, validActual, testScore, testPrediction, testActual
            
    def replicateAlpha(self):
        '''
        Reset kAlpha attribute and replicate currAlpha and store them to kAlpha dictionary.

        Returns
        -------
        None.

        '''
        self.kAlphas = {self.symbolList[0]: self.currAlpha}
        for symbol in self.symbolList[1:]:
            self.kAlphas[symbol] = copy.deepcopy(self.currAlpha)
            
    def reconstruct_kAlpha(self):
        self.replicateAlpha()
        for i, symbol in enumerate(self.symbolList):
            for node, valueList in self.OperandsValues.items():
                self.kAlphas[symbol].graph.nodes[node].updateValue(valueList[i])
                
    def last_check(self):
        self.currAlpha = self.bestFit[0]
        self.OperandsValues = self.bestFit[5]
        self.replicateAlpha()
        self.reconstruct_kAlpha()
        
        testScore, testPrediction, testActual = self.test_last_check()
        backtest(testActual, testPrediction, True, pctLongShort = self.pctLongShort)
    
        
    def resetOperandsValues(self):
        '''
        After evaluate 1 alpha, we need to reset the OperandsValues dictionary to evaluate the new currAlpha

        Returns
        -------
        None.

        '''
        self.OperandsValues = {}
        for node in self.currAlpha.graph.nodes.keys():
            self.OperandsValues[node] = [self.kAlphas[symbol].graph.nodes[node].value for symbol in self.symbolList]
            #self.kAlphas[symbol] -> alpha
            #self.kAlphas[symbol].graph -> alpha.graph
            #self.kAlphas[symbol].graph.nodes -> dictionary of nodes of graph
            #self.kAlphas[symbol].graph.nodes[node] -> class Scalar/Vector/Matrix of node
            #self.kAlphas[symbol].graph.nodes[node].value -> value: np.dnarray
    
    def runFirstWindow(self):
        '''
        fit the first window input to alpha.
        The purpose is to make sure every vector and matrix nodes have shape (!= None) and the operations are appropriate.
        While running this method, we also run executionOperation method which can handle the None value input (new Scalar/Vector/Matrix) 
        and delete all those operations which are not valid.
        
        Ex: after mutating, we have these operations:
            1. ['v3', 20, ['v1']] ~ v3 = 1/v1
            2. ['v2', 22, ['v2']] ~ v2 = |v3|
            while shape of v1 is 10, shape of v2 is 5 and shape of v3 is None.
            after 1st operation, shape of v3 is 10 -> 2nd operation is invalid -> delete 2nd operation
            -> method runFirstWindow() handle the shape of vector and matrix nodes.

        Returns
        -------
        None.

        '''
        cnt = 0
        while True and cnt < 10:
            cnt += 1
            self.replicateAlpha()
            self.resetOperandsValues()
            self.addM0(self.window)
            self.setup()
            self.predict()
            self.addS0(self.window)
            self.update()
            self.currAlpha.fillUndefinedOperands()
            self.prunning()
            Shapes = [node.shape for node in self.currAlpha.graph.nodes.values()]
            if None in Shapes:
                break
            
    def train(self):
        '''
        train the current alpha using train data

        Returns
        -------
        None.

        '''
        for i in list(np.random.choice(range(self.window, self.trainLength), size = (self.trainLength - self.window), replace= False)):
            #self.currAlpha.graph.show()
            self.addM0(i)
            self.setup()
            self.predict()
            self.addS0(i)
            self.update()
    
    def validate(self):
        '''
        evaluate and predict the current alpha using validation data

        Returns
        -------
        tuple
            fitness score, prediction, actual.

        '''
        fitnessScore = []
        MIScore = []
        validPrediction = []
        validActual = []
        for i in range(self.trainLength, self.trainLength + self.validLength):
            #self.currAlpha.graph.show()
            self.addM0(i)
            self.setup()
            self.predict()
            self.addS0(i)
            assert len(self.OperandsValues['s1']) == len(self.OperandsValues['s0']) == len(self.symbolList)
            fitnessScore.append(np.corrcoef(self.OperandsValues['s1'].copy(), self.OperandsValues['s0'].copy())[0,1])
            validPrediction.append(self.OperandsValues['s1'].copy())
            validActual.append(self.OperandsValues['s0'].copy())
            #validPrediction.append([copy.deepcopy(self.scaler[symbol][1]).inverse_transform(np.array(self.OperandsValues['s1'].copy()[j]).reshape(-1,1))[-1,0] for j,symbol in enumerate(self.symbolList)])
            #validActual.append([copy.deepcopy(self.scaler[symbol][1]).inverse_transform(np.array(self.OperandsValues['s0'].copy()[j]).reshape(-1,1))[-1,0] for j,symbol in enumerate(self.symbolList)])
            #fitnessScore.append(np.corrcoef(validPrediction[-1].copy(), validActual[-1].copy())[0,1])
        fitnessScore = sum(fitnessScore)/len(fitnessScore)#/np.std(fitnessScore)
        #if fitness score is nan -> s1 is constance -> set fitness score to the lowest value possible
        if np.isnan(fitnessScore): fitnessScore = -1
        try:
            for i in range(len(self.symbolList)):
                MIScore.append(calc_MI([validPrediction[j][i] for j in range(len(validPrediction))], [validActual[j][i] for j in range(len(validActual))]))
                
            MIScore = sum(MIScore)/len(MIScore)
        except: MIScore = 0
        
        return fitnessScore, np.array(validPrediction, dtype = np.float64), np.array(validActual, dtype = np.float64)
        
    def test(self):
        '''
        evaluate and predict the current alpha using test data

        Returns
        -------
        tuple
            test score, prediction, actual.

        '''
        testScore = []
        MIScore = []
        testPrediction = []
        testActual = []
        for i in range(self.trainLength + self.validLength, self.trainLength + self.validLength + self.testLength):
            #self.currAlpha.graph.show()
            self.addM0(i)
            self.setup()
            self.predict()
            self.addS0(i)
            assert len(self.OperandsValues['s1']) == len(self.OperandsValues['s0']) == len(self.symbolList)
            testScore.append(np.corrcoef(self.OperandsValues['s1'].copy(), self.OperandsValues['s0'].copy())[0,1])
            testPrediction.append(self.OperandsValues['s1'].copy())
            testActual.append(self.OperandsValues['s0'].copy())
            #testPrediction.append([copy.deepcopy(self.scaler[symbol][1]).inverse_transform(np.array(self.OperandsValues['s1'].copy()[j]).reshape(-1,1))[-1,0] for j,symbol in enumerate(self.symbolList)])
            #testActual.append([copy.deepcopy(self.scaler[symbol][1]).inverse_transform(np.array(self.OperandsValues['s0'].copy()[j]).reshape(-1,1))[-1,0] for j,symbol in enumerate(self.symbolList)])
            #testScore.append(np.corrcoef(testPrediction[-1].copy(), testActual[-1].copy())[0,1])
        testScore = sum(testScore)/len(testScore)#/np.std(testScore)
        #if test score is nan -> s1 is constance -> set test score to the lowest value possible
        if np.isnan(testScore): testScore = -1
        try:
            for i in range(len(self.symbolList)):
                MIScore.append(calc_MI([testPrediction[j][i] for j in range(len(testPrediction))], [testActual[j][i] for j in range(len(testActual))]))
            
            MIScore = sum(MIScore)/len(MIScore)
        except: MIScore = 0
        
        return  testScore, np.array(testPrediction, dtype = np.float64), np.array(testActual, dtype = np.float64)
    
    def test_last_check(self):
        '''
        evaluate and predict the current alpha using test data

        Returns
        -------
        tuple
            test score, prediction, actual.

        '''
        testScore = []
        testPrediction = []
        testActual = []
        for i in range(self.trainLength + self.validLength + self.testLength, self.dataLength - 1):
            #self.currAlpha.graph.show()
            self.addM0(i)
            self.setup()
            self.predict()
            self.addS0(i)
            assert len(self.OperandsValues['s1']) == len(self.OperandsValues['s0']) == len(self.symbolList)
            testScore.append(np.corrcoef(self.OperandsValues['s1'].copy(), self.OperandsValues['s0'].copy())[0,1])
            testPrediction.append(self.OperandsValues['s1'].copy())
            testActual.append(self.OperandsValues['s0'].copy())
        testScore = sum(testScore)/len(testScore)
        if np.isnan(testScore): testScore = -1
        return  testScore, np.array(testPrediction, dtype = np.float64), np.array(testActual, dtype = np.float64)
    
    def setup(self):
        '''
        execute all operation in setup step for all Alphas of each symbol

        Returns
        -------
        None.

        '''
        for Operation in self.currAlpha.graph.setupOPs:
            self.executeOperation(Operation)
    
    def addM0(self, i: int):
        '''
        add window input at the date i-th to operand m0

        Parameters
        ----------
        i : int
            index of the current date to fit window.

        Returns
        -------
        None.

        '''
        for j in range(len(self.symbolList)):
            symbol = self.symbolList[j]
            
            X, y = self.createWindow(symbol, i)
            
            self.kAlphas[symbol].graph.addM0(X)
            self.OperandsValues['m0'][j] = X
    
    def predict(self):
        '''
        execute all operation in predict step for all Alphas of each symbol

        Returns
        -------
        None.

        '''
        for Operation in self.currAlpha.graph.predictOPs:
            self.executeOperation(Operation)
    
    def addS0(self, i:int):
        '''
        add actual returns output at the date i-th to operand s0

        Parameters
        ----------
        i : int
            index of the current date to fit output.

        Returns
        -------
        None.

        '''
        self.OperandsValues['s0'] = [0]*len(self.symbolList)
        
        for j in range(len(self.symbolList)):
            symbol = self.symbolList[j]
            
            X, y = self.createWindow(symbol, i)
            
            self.kAlphas[symbol].graph.addS0(y)
            self.OperandsValues['s0'][j] = y
    
    def update(self):
        '''
        execute all operation in update step for all Alphas of each symbol

        Returns
        -------
        None.

        '''
        for Operation in self.currAlpha.graph.updateOPs:
            self.executeOperation(Operation)
        
    def prunning(self):
        # prunning the current Alpha
        self.currAlpha.prunning()
        #self.currAlpha.graph.show()
    
    def fingerprint(self):
        #create fingerprint of the current Alpha
        return self.currAlpha.fingerprint()
    
    def summaryAlpha(self, show: bool = True):
        '''
        print out basic metric after evaluating the current Alpha

        Parameters
        ----------
        show : bool, optional
            If True, print out the metric. If False, evaluate only. The default is True.

        Returns
        -------
        fitnessScore : float
        dailyReturns : np.ndarray
            The array of portfolio returns.
        annualizedReturns : float
        sharpe : float

        '''
        fitnessScore, validPrediction, validActual, testScore, testPrediction, testActual = self.evaluate()
        '''returns = np.zeros(testActual.shape)
        Prediction = np.zeros(testPrediction.shape)
        for i, symbol in enumerate(self.symbolList):
            scaler_label = self.scaler[i][1]
            returns[:, i] = scaler_label.inverse_transform(np.array(testActual[:, i]).reshape(-1, 1)).reshape(testActual.shape[0],)
            try:
                Prediction[:, i] = scaler_label.inverse_transform(np.array(testPrediction[:, i]).reshape(-1, 1)).reshape(testPrediction.shape[0],)
            except:
                Prediction = testPrediction'''
        #for i, symbol in enumerate(self.symbolList):
            #check invert transform correct or not
            #assert (testActual[:, i] - np.array(self.data[symbol]['close_return'].iloc[self.trainLength + self.validLength:self.dataLength-1]).reshape(testActual.shape[0],) < 0.00001).all()
        if show:
            print('========================================')
            self.currAlpha.graph.show()
            print('VALIDATION FITNESS:', fitnessScore)
            print('TEST FITNESS      :', testScore)
            dailyReturns, annualizedReturns, sharpe, ultilization, returnDetails = backtest(testActual, testPrediction, show, self.pctLongShort)
            print('========================================')
        else:
            dailyReturns, annualizedReturns, sharpe, ultilization, returnDetails = backtest(testActual, testPrediction, show, self.pctLongShort)
        return fitnessScore, dailyReturns, annualizedReturns, sharpe, testScore, ultilization, returnDetails, validActual, validPrediction
    
    def summaryBestFit(self):
        '''
        print out basic metric of the best fitness Alpha since the first time

        Returns
        -------
        None.

        '''
        print('=====================================================')
        self.bestFit[0].graph.show()
        print('FITNESS SCORE     :', self.bestFit[1])
        print('ANNUALIZED RETURNS:', self.bestFit[3])
        print('SHARPE            :', self.bestFit[4])
        plt.figure(figsize = (6,4))
        plt.plot(self.bestFit[2])
        plt.title('Cumulative Returns')
        plt.ylabel('Returns')
        plt.show()
        print('=====================================================')
    
    def getBestFit(self, group: list):
        '''
        return the best fitness score of alpha in attribute tournament and point currAlpha to that alpha

        Returns
        -------
        bestFit : float
            the highest fitness score in tournament.

        '''
        bestFit = -10000
        for alpha in group:
            fitnessScore = self.fitnessScore[alpha.fingerprint()]
            if fitnessScore >= bestFit:
                bestFit = fitnessScore
                bestAlpha = alpha
        return bestFit, bestAlpha
    
    def mutate(self):
        #mutate the currAlpha
        self.currAlpha.mutate()
    
    def pickTournament(self):
        #    randomly choose tournamentLength alphas in population and bestfit alpha
        # -> this method facilitate to choose best fit alpha ever (even if that alpha is not in population)
        self.tournament = np.random.choice(self.population, replace = False, size = self.tournamentLength)
    
    def extractAlpha(self, name: str = 'abc'):
        #handle int32 and float32 to serialize to json
        for key, value in self.bestFit[5].items():
            value = np.array(value, dtype = np.float64).tolist()
            self.bestFit[5][key] = value
        
        for operation in self.bestFit[0].graph.updateOPs + self.bestFit[0].graph.predictOPs + self.bestFit[0].graph.updateOPs:
            for i, element in enumerate(operation):
                if isinstance(element, np.int32) or isinstance(element, np.int64):
                    operation[i] = int(element)
                elif isinstance(element, np.float32):
                    operation[i] = float(element)
        
        # post API
        url = 'http://13.113.253.201/api/alpha'

        headers = {'Token': 'q0hcdABLUhGAzW3j',
                'Content-Type': 'application/json'}
        
        body = json.dumps({ 'symbolList': self.symbolList,
         'window' : self.window,
         'nodes' : list(self.bestFit[0].graph.nodes.keys()),
         'setupOPs' : self.bestFit[0].graph.setupOPs,
         'predictOPs' : self.bestFit[0].graph.predictOPs,
         'updateOPs' : self.bestFit[0].graph.updateOPs,
         'operandsValues': self.bestFit[5],
         'name' : name })
        
        requests.post(url = url, data = body, headers = headers)
        
    def executeOperation(self, Operation: list):
        '''
        This function execute Operation by:
            1. handle None value/shape operands (input or output)
            2. check shape/value valid or not. If not valid -> delete the Operation in all alphas in kAlpha
            3. if valid, update value of output node of all alphas in kAlpha
                        and update value of output node of OperandsValues dictionary

        Parameters
        ----------
        Operation : list
            [Output: str, OP: int, Inputs: list].

        Returns
        -------
        None.

        '''
        Output, op, Inputs = Operation
        if op in [1, 2]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalar1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalar2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 0
                if scalar1.value is None:
                   scalar1.updateValue(0)
                   self.OperandsValues[Inputs[0]][i] = 0
                if scalar2.value is None:
                   scalar2.updateValue(0)
                   self.OperandsValues[Inputs[1]][i] = 0
                
                if op == 1:
                    outputValue = OP1(scalar1, scalar2)
                elif op == 2:
                    outputValue = OP2(scalar1, scalar2)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 3:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalar1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalar2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalar1.value is None:
                   scalar1.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                if scalar2.value is None:
                   scalar2.updateValue(1)
                   self.OperandsValues[Inputs[1]][i] = 1
                
                outputValue = OP3(scalar1, scalar2)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 4:
            if (np.array(self.OperandsValues[Inputs[1]])==None).any():
                pass
            elif (np.round(np.array(self.OperandsValues[Inputs[1]]), 6) == 0).any():
                for i in range(len(self.symbolList)):
                    #print('DEL:', Operation)
                    symbol = self.symbolList[i]
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                return
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalar1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalar2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalar1.value is None:
                   scalar1.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                if scalar2.value is None:
                   scalar2.updateValue(1)
                   self.OperandsValues[Inputs[1]][i] = 1
                   
                outputValue = OP4(scalar1, scalar2)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [5, 13, 15]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None:
                   scalarInput.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                
                if op == 5:
                    outputValue = OP5(scalarInput)
                elif op == 13:
                    outputValue = OP13(scalarInput)
                elif op == 15:
                    outputValue = OP15(scalarInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 6:
            if (np.array(self.OperandsValues[Inputs[0]]) == None).any():
                pass
            elif (np.round(np.array(self.OperandsValues[Inputs[0]]), 6) == 0).any():
                for i in range(len(self.symbolList)):
                    #print('DEL:', Operation)
                    symbol = self.symbolList[i]
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                return
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None:
                   scalarInput.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                
                outputValue = OP6(scalarInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
                    
        
        if op in [7, 8, 9, 12]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 0
                if scalarInput.value is None:
                   scalarInput.updateValue(0)
                   self.OperandsValues[Inputs[0]][i] = 0
                
                if op == 7:
                    outputValue = OP7(scalarInput)
                elif op == 8:
                    outputValue = OP8(scalarInput)
                elif op == 9:
                    outputValue = OP9(scalarInput)
                elif op == 12:
                    outputValue = OP12(scalarInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
                
        
        if op in [10, 11]:
            if (np.array(self.OperandsValues[Inputs[0]]) == None).any():
                pass
            elif (abs(np.round(np.array(self.OperandsValues[Inputs[0]]), 6)) > 1).any():
                for i in range(len(self.symbolList)):
                    #print('DEL:', Operation)
                    symbol = self.symbolList[i]
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                return
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None:
                   scalarInput.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                
                if op == 10:
                    outputValue = OP10(scalarInput)
                elif op == 11:
                    outputValue = OP11(scalarInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
                
        
        if op == 14:
            if (np.array(self.OperandsValues[Inputs[0]]) == None).any():
                pass
            elif (np.round(np.array(self.OperandsValues[Inputs[0]]), 6) <= 0).any():
                for i in range(len(self.symbolList)):
                    #print('DEL:', Operation)
                    symbol = self.symbolList[i]
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                return
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None:
                   scalarInput.updateValue(1)
                   self.OperandsValues[Inputs[0]][i] = 1
                   
                outputValue = OP14(scalarInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
                
        
        if op == 16:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.value is None:
                    if vectorOutput.shape is None:
                        vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    else:
                        vectorInput.updateValue(np.ones(vectorOutput.shape)) # len = len of output
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                
                #if input and output shape do not match -> delete this operation
                if vectorInput.shape is not None and vectorOutput.shape is not None and vectorInput.shape != vectorOutput.shape:
                    #print('DEL:', Operation, vectorInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                
                else:
                    outputValue = OP16(vectorInput)
                    
                    vectorOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
            
        
        if op in [17, 30, 38]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.value is None:
                    if matrixOutput.shape is None:
                        matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    else:
                        matrixInput.updateValue(np.ones(shape = matrixOutput.shape)) # shape = shape of output
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                
                #if input and output shape do not match -> delete this operation
                if matrixInput.shape is not None and matrixOutput.shape is not None and matrixInput.shape != matrixOutput.shape:
                    #print('DEL:', Operation, matrixInput.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    
                else:
                    if op == 17:
                        outputValue = OP17(matrixInput)
                    elif op == 30:
                        if (np.round(matrixInput.value, 6) != 0).all():
                            outputValue = OP30(matrixInput)
                        else:
                            #print('DEL:', Operation)
                            if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                            if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                            if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                            break
                    elif op == 38:
                        outputValue = OP38(matrixInput)
                    
                    matrixOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
            
        
        if op == 18:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None: 
                    scalarInput.updateValue(1)
                    self.OperandsValues[Inputs[0]][i] = 1
                if vectorInput.value is None:
                    if vectorOutput.shape is None:
                        vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    else:
                        vectorInput.updateValue(np.ones(vectorOutput.shape)) # len = len of output
                    self.OperandsValues[Inputs[1]][i] = vectorInput.value
                    
                #if input and output shape do not match -> delete this operation
                if vectorInput.shape is not None and vectorOutput.shape is not None and vectorInput.shape != vectorOutput.shape:
                    #print('DEL:', Operation, vectorInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                
                else:
                    outputValue = OP18(scalarInput, vectorInput)
                    
                    vectorOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
                    
        
        if op == 19:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                integerInput = Inputs[1]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None: 
                    scalarInput.updateValue(1)
                    self.OperandsValues[Inputs[0]][i] = 1
                #if input and output shape do not match -> delete this operation
                if vectorOutput.shape is not None and vectorOutput.shape != integerInput:
                    #print('DEL:', Operation, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                else:
                    outputValue = OP19(scalarInput, integerInput)
                    
                    vectorOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
        
        
        if op == 20:
            if (np.array(self.OperandsValues[Inputs[0]])==None).any():
                pass
            elif (np.round(np.array(self.OperandsValues[Inputs[0]]), 6) == 0).any():
                for i in range(len(self.symbolList)):
                    #print('DEL:', Operation)
                    symbol = self.symbolList[i]
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                return
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.value is None:
                    if vectorOutput.shape is None:
                        vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    else:
                        vectorInput.updateValue(np.ones(vectorOutput.shape)) # len = len of output
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                    
                #if input and output shape do not match -> delete this operation
                if vectorInput.shape is not None and vectorOutput.shape is not None and vectorInput.shape != vectorOutput.shape:
                    #print('DEL:', Operation, vectorInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                
                else:
                    outputValue = OP20(vectorInput)
                    
                    vectorOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
        
        
        if op in [21, 50, 54]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.value is None:
                    vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                    
                if op == 21:
                    outputValue = OP21(vectorInput)
                elif op == 50:
                    outputValue = OP50(vectorInput)
                elif op == 54:
                    outputValue = OP54(vectorInput)
                    
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 22:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.value is None:
                    if vectorOutput.shape is None:
                        vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    else:
                        vectorInput.updateValue(np.ones(vectorOutput.shape)) # len = len of output
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                #if input and output shape do not match -> delete this operation
                if vectorInput.shape is not None and vectorOutput.shape is not None and vectorInput.shape != vectorOutput.shape:
                    #print('DEL:', Operation, vectorInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                
                else:
                    outputValue = OP22(vectorInput)
                    
                    vectorOutput.updateValue(outputValue)
                    self.OperandsValues[Output][i] = outputValue
                    
        
        if op in [23, 24, 25, 26, 45, 48]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorInput2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput1.shape is None and vectorInput2.shape is None and vectorOutput.shape is None:
                    vectorInput1.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                    vectorInput2.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is None and vectorOutput.shape is not None:
                    vectorInput1.updateValue(np.ones(vectorOutput.shape))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                    vectorInput2.updateValue(np.ones(vectorOutput.shape))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is not None and vectorOutput.shape is None:
                    vectorInput1.updateValue(np.ones(vectorInput2.shape))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                elif vectorInput1.shape is not None and vectorInput2.shape is None and vectorOutput.shape is None:
                    vectorInput2.updateValue(np.ones(vectorInput1.shape))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is not None and vectorOutput.shape is not None and vectorInput2.shape == vectorOutput.shape:
                    vectorInput1.updateValue(np.ones(vectorOutput.shape))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                elif vectorInput1.shape is not None and vectorInput2.shape is None and vectorOutput.shape is not None and vectorInput1.shape == vectorOutput.shape:
                    vectorInput2.updateValue(np.ones(vectorOutput.shape))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is not None and vectorInput2.shape is not None and vectorOutput.shape is None and vectorInput1.shape == vectorInput2.shape:
                    pass
                elif vectorInput1.shape is not None and vectorInput2.shape is not None and vectorOutput.shape is not None and vectorInput1.shape == vectorInput2.shape == vectorOutput.shape:
                    pass
                else:
                    #print('DEL:', Operation, vectorInput1.shape, vectorInput2.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                if op == 23:
                    outputValue = OP23(vectorInput1, vectorInput2)
                elif op == 24:
                    outputValue = OP24(vectorInput1, vectorInput2)    
                elif op == 25:
                    outputValue = OP25(vectorInput1, vectorInput2)
                elif op == 26:
                    if (np.round(vectorInput2.value, 6) != 0).all():
                        outputValue = OP26(vectorInput1, vectorInput2)
                    else:
                        #print('DEL:', Operation, vectorInput1.shape, vectorInput2.shape, vectorOutput.shape)
                        if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                        if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                        if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                        break
                elif op == 45:
                    outputValue = OP45(vectorInput1, vectorInput2)
                elif op == 48:
                    outputValue = OP48(vectorInput1, vectorInput2)
                    
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
    
    
        if op == 27:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorInput2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput1.shape is None and vectorInput2.shape is None:
                    vectorInput1.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                    vectorInput2.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is not None:
                    vectorInput1.updateValue(np.ones(vectorInput2.shape))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                elif vectorInput1.shape is not None and vectorInput2.shape is None:
                    vectorInput2.updateValue(np.ones(vectorInput1.shape))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is not None and vectorInput2.shape is not None and vectorInput1.shape == vectorInput2.shape:
                    pass
                else:
                    #print('DEL:', Operation, vectorInput1.shape, vectorInput2.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP27(vectorInput1, vectorInput2)
                    
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 28:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorInput2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput1.shape is None and vectorInput2.shape is None and matrixOutput.shape is None:
                    vectorInput1.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                    vectorInput2.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is None and matrixOutput.shape is not None:
                    vectorInput1.updateValue(np.ones(matrixOutput.shape[0]))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                    vectorInput2.updateValue(np.ones(matrixOutput.shape[1]))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is not None and matrixOutput.shape is None:
                    vectorInput1.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                elif vectorInput1.shape is not None and vectorInput2.shape is None and matrixOutput.shape is None:
                    vectorInput2.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is None and vectorInput2.shape is not None and matrixOutput.shape is not None and vectorInput2.shape == matrixOutput.shape[1]:
                    vectorInput1.updateValue(np.ones(matrixOutput.shape[0]))
                    self.OperandsValues[Inputs[0]][i] = vectorInput1.value
                elif vectorInput1.shape is not None and vectorInput2.shape is None and matrixOutput.shape is not None and vectorInput1.shape == matrixOutput.shape[0]:
                    vectorInput2.updateValue(np.ones(matrixOutput.shape[1]))
                    self.OperandsValues[Inputs[1]][i] = vectorInput2.value
                elif vectorInput1.shape is not None and vectorInput2.shape is not None and matrixOutput.shape is None:
                    pass
                elif vectorInput1.shape is not None and vectorInput2.shape is not None and matrixOutput.shape is not None and matrixOutput.shape == (vectorInput1.shape, vectorInput2.shape):
                    pass
                else:
                    #print('DEL:', Operation, vectorInput1.shape, vectorInput2.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP28(vectorInput1, vectorInput2)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 29:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if scalarInput.value is None: 
                    scalarInput.updateValue(1)
                    self.OperandsValues[Inputs[0]][i] = 1
                if matrixInput.shape is None and matrixOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[1]][i] = matrixInput.value
                elif matrixInput.shape is None and matrixOutput.shape is not None:
                    matrixInput.updateValue(np.ones(shape = matrixOutput.shape))
                    self.OperandsValues[Inputs[1]][i] = matrixInput.value
                elif matrixInput.shape is not None and matrixOutput.shape is None: 
                    pass
                elif matrixInput.shape is not None and matrixOutput.shape is not None and matrixInput.shape == matrixOutput.shape:
                    pass
                else:
                    #print('DEL:', Operation, matrixInput.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP29(scalarInput, matrixInput)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 31:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.shape is None and vectorInput.shape is None and vectorOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                    vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput.value
                elif matrixInput.shape is None and vectorInput.shape is None and vectorOutput.shape is not None:
                    vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[1]][i] = vectorInput.value
                    matrixInput.updateValue(np.ones(shape = (vectorOutput.shape, vectorInput.shape)))
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is None and vectorInput.shape is not None and vectorOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, vectorInput.shape))) # len = window
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is not None and vectorInput.shape is None and vectorOutput.shape is None:
                    vectorInput.updateValue(np.ones(matrixInput.shape[1]))
                    self.OperandsValues[Inputs[1]][i] = vectorInput.value
                elif matrixInput.shape is None and vectorInput.shape is not None and vectorOutput.shape is not None:
                    matrixInput.updateValue(np.ones(shape = (vectorOutput.shape, vectorInput.shape)))
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is not None and vectorInput.shape is None and vectorOutput.shape is not None and matrixInput.shape[0] == vectorOutput.shape:
                    vectorInput.updateValue(np.ones(matrixInput.shape[1]))
                    self.OperandsValues[Inputs[1]][i] = vectorInput.value
                elif matrixInput.shape is not None and vectorInput.shape is not None and vectorOutput.shape is None and matrixInput.shape[1] == vectorInput.shape:
                    pass
                elif matrixInput.shape is not None and vectorInput.shape is not None and vectorOutput.shape is not None and matrixInput.shape == (vectorOutput.shape, vectorInput.shape):
                    pass
                else:
                    #print('DEL:', Operation, matrixInput.shape, vectorInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP31(matrixInput, vectorInput)
                    
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 32:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                integerInput = Inputs[1]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.shape is None and matrixOutput.shape is None:
                    vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                elif vectorInput.shape is None and matrixOutput.shape is not None and matrixOutput.shape[0] == integerInput:
                    vectorInput.updateValue(np.ones(matrixOutput.shape[1]))
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                elif vectorInput.shape is not None and matrixOutput.shape is None:
                    pass
                elif vectorInput.shape is not None and matrixOutput.shape is not None and matrixOutput.shape == (integerInput, vectorInput.shape):
                    pass
                else:
                    #print('DEL:', Operation, vectorInput.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP32(vectorInput, integerInput)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 33:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                vectorInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                integerInput = Inputs[1]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if vectorInput.shape is None and matrixOutput.shape is None:
                    vectorInput.updateValue(np.ones(len(self.featuresList))) #len = number of features
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                elif vectorInput.shape is None and matrixOutput.shape is not None and matrixOutput.shape[1] == integerInput:
                    vectorInput.updateValue(np.ones(matrixOutput.shape[0]))
                    self.OperandsValues[Inputs[0]][i] = vectorInput.value
                elif vectorInput.shape is not None and matrixOutput.shape is None:
                    pass
                elif vectorInput.shape is not None and matrixOutput.shape is not None and matrixOutput.shape == (vectorInput.shape, integerInput):
                    pass
                else:
                    #print('DEL:', Operation, vectorInput.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP33(vectorInput, integerInput)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [34, 51, 55]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                
                if op == 34:
                    outputValue = OP34(matrixInput)
                elif op == 51:
                    outputValue = OP51(matrixInput)
                elif op == 55:
                    outputValue = OP55(matrixInput)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
                
        if op in [35, 52, 53]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.shape is None and vectorOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is None and vectorOutput.shape is not None:
                    matrixInput.updateValue(np.ones(shape = (vectorOutput.shape, len(self.featuresList))))
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is not None and vectorOutput.shape is None:
                    pass
                elif matrixInput.shape is not None and vectorOutput.shape is not None and vectorOutput.shape == matrixInput.shape[0]:
                    pass
                else:
                    #print('DEL:', Operation, matrixInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                if op == 35:
                    outputValue = OP35(matrixInput)
                elif op == 52:
                    outputValue = OP52(matrixInput)
                elif op == 53:
                    outputValue = OP53(matrixInput)
                    
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 36:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.shape is None and vectorOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is None and vectorOutput.shape is not None:
                    matrixInput.updateValue(np.ones(shape = (len(self.featuresList), vectorOutput.shape)))
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is not None and vectorOutput.shape is None:
                    pass
                elif matrixInput.shape is not None and vectorOutput.shape is not None and vectorOutput.shape == matrixInput.shape[1]:
                    pass
                else:
                    #print('DEL:', Operation, matrixInput.shape, vectorOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP36(matrixInput)
                    
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 37:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput.shape is None and matrixOutput.shape is None:
                    matrixInput.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) # shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is None and matrixOutput.shape is not None:
                    matrixInput.updateValue(np.ones(shape = (matrixOutput.shape[1], matrixOutput.shape[0])))
                    self.OperandsValues[Inputs[0]][i] = matrixInput.value
                elif matrixInput.shape is not None and matrixOutput.shape is None:
                    pass
                elif matrixInput.shape is not None and matrixOutput.shape is not None and matrixInput.shape == (matrixOutput.shape[1], matrixOutput.shape[0]):
                    pass
                else:
                    #print('DEL:', Operation, matrixInput.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                outputValue = OP37(matrixInput)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [39, 40, 41, 42, 46, 49]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                matrixInput2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput1.shape is None and matrixInput2.shape is None and matrixOutput.shape is None:
                    matrixInput1.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) #shape = shape of m0
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                    matrixInput2.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) #shape = shape of m0
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is None and matrixOutput.shape is not None:
                    matrixInput1.updateValue(np.ones(shape = matrixOutput.shape))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                    matrixInput2.updateValue(np.ones(shape = matrixOutput.shape))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is not None and matrixOutput.shape is None:
                    matrixInput1.updateValue(np.ones(shape = matrixInput2.shape))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                elif matrixInput1.shape is not None and matrixInput2.shape is None and matrixOutput.shape is None:
                    matrixInput2.updateValue(np.ones(shape = matrixInput1.shape))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is not None and matrixOutput.shape is not None and matrixInput2.shape == matrixOutput.shape:
                    matrixInput1.updateValue(np.ones(shape = matrixOutput.shape))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                elif matrixInput1.shape is not None and matrixInput2.shape is None and matrixOutput.shape is not None and matrixInput1.shape == matrixOutput.shape:
                    matrixInput2.updateValue(np.ones(shape = matrixOutput.shape))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is not None and matrixInput2.shape is not None and matrixOutput.shape is None and matrixInput1.shape == matrixInput2.shape:
                    pass
                elif matrixInput1.shape is not None and matrixInput2.shape is not None and matrixOutput.shape is not None and matrixInput1.shape == matrixInput2.shape == matrixOutput.shape:
                    pass
                else:
                    #print('DEL:', Operation, matrixInput1.shape, matrixInput2.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
                
                if op == 39:
                    outputValue = OP39(matrixInput1, matrixInput2)
                elif op == 40:
                    outputValue = OP40(matrixInput1, matrixInput2)    
                elif op == 41:
                    outputValue = OP41(matrixInput1, matrixInput2)
                elif op == 42:
                    if (np.round(matrixInput2.value,6) != 0).all():
                        outputValue = OP42(matrixInput1, matrixInput2)
                    else:
                        #print('DEL:', Operation, matrixInput1.shape, matrixInput2.shape, matrixOutput.shape)
                        if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                        if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                        if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                        break
                elif op == 46:
                    outputValue = OP46(matrixInput1, matrixInput2)
                elif op == 49:
                    outputValue = OP49(matrixInput1, matrixInput2)
                    
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 43:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                matrixInput1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                matrixInput2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 1
                if matrixInput1.shape is None and matrixInput2.shape is None and matrixOutput.shape is None:
                    matrixInput1.updateValue(np.ones(shape = (len(self.featuresList), self.window))) #shape = shape of m0.T
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                    matrixInput2.updateValue(np.ones(shape = (self.window, len(self.featuresList)))) #shape = shape of m0
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is None and matrixOutput.shape is not None:
                    randomLength = np.random.randint(2, 20)
                    matrixInput1.updateValue(np.ones(shape = (matrixOutput.shape[0], randomLength)))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                    matrixInput2.updateValue(np.ones(shape = (randomLength, matrixOutput.shape[1])))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is not None and matrixOutput.shape is None:
                    randomLength = np.random.randint(2, 20)
                    matrixInput1.updateValue(np.ones(shape = (randomLength, matrixInput2.shape[0])))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                elif matrixInput1.shape is not None and matrixInput2.shape is None and matrixOutput.shape is None:
                    randomLength = np.random.randint(2, 20)
                    matrixInput2.updateValue(np.ones(shape = (matrixInput1.shape[1], randomLength)))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is None and matrixInput2.shape is not None and matrixOutput.shape is not None and matrixInput2.shape[1] == matrixOutput.shape[1]:
                    matrixInput1.updateValue(np.ones(shape = (matrixOutput.shape[0], matrixInput2.shape[0])))
                    self.OperandsValues[Inputs[0]][i] = matrixInput1.value
                elif matrixInput1.shape is not None and matrixInput2.shape is None and matrixOutput.shape is not None and matrixInput1.shape[0] == matrixOutput.shape[0]:
                    matrixInput2.updateValue(np.ones(shape = (matrixInput1.shape[1], matrixOutput.shape[1])))
                    self.OperandsValues[Inputs[1]][i] = matrixInput2.value
                elif matrixInput1.shape is not None and matrixInput2.shape is not None and matrixOutput.shape is None and matrixInput1.shape[1] == matrixInput2.shape[0]:
                    pass
                elif matrixInput1.shape is not None and matrixInput2.shape is not None and matrixOutput.shape is not None and matrixInput1.shape[1] == matrixInput2.shape[0] and matrixInput1.shape[0] == matrixOutput.shape[0] and matrixInput2.shape[1] == matrixOutput.shape[1]:
                    pass
                else:
                    #print('DEL:', Operation, matrixInput1.shape, matrixInput2.shape, matrixOutput.shape)
                    if Operation in self.kAlphas[symbol].graph.setupOPs: self.kAlphas[symbol].graph.setupOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.predictOPs: self.kAlphas[symbol].graph.predictOPs.remove(Operation)
                    if Operation in self.kAlphas[symbol].graph.updateOPs: self.kAlphas[symbol].graph.updateOPs.remove(Operation)
                    continue
            
                outputValue = OP43(matrixInput1, matrixInput2)
                
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [44, 47]:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                
                scalar1 = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                scalar2 = self.kAlphas[symbol].graph.nodes[Inputs[1]]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                #make sure input value is not None. If it is None -> initiate to be 0
                if scalar1.value is None:
                   scalar1.updateValue(0)
                   self.OperandsValues[Inputs[0]][i] = 0
                if scalar2.value is None:
                   scalar2.updateValue(0)
                   self.OperandsValues[Inputs[1]][i] = 0
                
                if op == 44:
                    outputValue = OP44(scalar1, scalar2)
                elif op == 47:
                    outputValue = OP47(scalar1, scalar2)
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 56:
            outputValue = OP56(Inputs[0])
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 57:
            outputValue = OP57(Inputs[0], Inputs[1])
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op == 58:
            outputValue = OP58(Inputs[0], Inputs[1], Inputs[2])
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [59, 62]:
            if op == 59:
                outputValue = OP59(Inputs[0], Inputs[1])
            elif op == 62:
                outputValue = OP62(Inputs[0], Inputs[1])
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                scalarOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        if op in [60, 63]:
            if op == 60:
                outputValue = OP60(Inputs[0], Inputs[1], Inputs[2])
            elif op == 63:
                outputValue = OP63(Inputs[0], Inputs[1], Inputs[2])
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                vectorOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                vectorOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        
        if op in [61, 64]:
            if op == 61:
                outputValue = OP61(Inputs[0], Inputs[1], Inputs[2], Inputs[3])
            elif op == 64:
                outputValue = OP64(Inputs[0], Inputs[1], Inputs[2], Inputs[3])
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                matrixOutput = self.kAlphas[symbol].graph.nodes[Output]
                
                matrixOutput.updateValue(outputValue)
                self.OperandsValues[Output][i] = outputValue
        
        if op == 65:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                
                if scalarInput.value is None:
                    scalarInput.updateValue(1)
                self.OperandsValues[Inputs[0]][i] = scalarInput.value
                   
            df = pd.DataFrame({'Scalar':self.OperandsValues[Inputs[0]]})
            self.OperandsValues[Output] = OP65(df)
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                scalarOutput.updateValue(self.OperandsValues[Output][i])
                
        if op == 66:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                
                if scalarInput.value is None:
                    scalarInput.updateValue(1)
                self.OperandsValues[Inputs[0]][i] = scalarInput.value
                   
            df = pd.DataFrame({'Scalar':self.OperandsValues[Inputs[0]], 'Industry': [0 for symbol in self.symbolList]})
            self.OperandsValues[Output] = OP66(df)
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                scalarOutput.updateValue(self.OperandsValues[Output][i])
        
        if op == 67:
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarInput = self.kAlphas[symbol].graph.nodes[Inputs[0]]
                
                if scalarInput.value is None:
                    scalarInput.updateValue(1)
                self.OperandsValues[Inputs[0]][i] = scalarInput.value
                   
            df = pd.DataFrame({'Scalar':self.OperandsValues[Inputs[0]], 'Industry': [0 for symbol in self.symbolList]})
            self.OperandsValues[Output] = OP67(df)
            
            for i in range(len(self.symbolList)):
                symbol = self.symbolList[i]
                scalarOutput = self.kAlphas[symbol].graph.nodes[Output]
                scalarOutput.updateValue(self.OperandsValues[Output][i])

def calc_MI(x, y):
    bins = min(int(np.sqrt(len(x))), len(np.unique(x)), len(np.unique(y)))
    try:
        c_xy = np.histogram2d(x, y, bins)[0]
        g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
        mi = 0.5 * g / c_xy.sum()
        
        #mi /= np.linalg.norm(np.array(x)-np.array(y))
    except:
        mi = 0
    return mi

if __name__ == '__main__':
    x = AlphaEvolve()
    x.run()