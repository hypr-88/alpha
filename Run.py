from AlphaEvolve import AlphaEvolve
from Graph import Graph
from Operands import *
if __name__ == '__main__':
    '''
    name: name of alpha that will be saved on the server to backtest on the terminal
    mutateProb: probability to mutate. If Binomial(1, mutateProb) = 0, then it does not mutate setup/predict/update
    population: length of population
    tournament: length of tournament
    window: length of input matrix
    numNewAlphaPerMutation: number of alphas mutated every step
    trainRatio: ratio of training data in dataset
    validRatio: ratio of test data in dataset
    TimeBudget: (Days, Hours, Minutes, Seconds)
    maxNumNodes: maximum number of nodes in each graph
    maxLenShapeNode: maximum length of every vector or matrix
    addProb: probability of mutation by adding
    delProb: probability of mutation by removing
    changeProb: probability of mutation by changing
    '''
    a = Graph()
    a.addNodes(Scalar(1))
    a.addSetupOPs(0, 's2', 56, [1])
    a.addPredictOPs(0, 's1', 3, ['s0', 's2'])
    x = AlphaEvolve(graph = a, name = '4symbols_005', mutateProb = 0.9, population = 50, tournament = 10, window = 30, 
                    numNewAlphaPerMutation = 15, trainRatio = 0.7, validRatio = 0.15, TimeBudget = (2, 0, 0, 0), 
                    maxNumNodes = (50, 200, 250), maxLenShapeNode = 50, addProb = 0.5, delProb = 0.1, changeProb = 0.5, frequency = '1D')
    x.run()