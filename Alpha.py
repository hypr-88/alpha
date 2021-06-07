import numpy as np
import copy
from Operands import Scalar, Vector, Matrix

from Graph import Graph
class Alpha():
    '''
    Class used to present an alpha
        ...
    Attributes
    ----------
    
    graph : Graph
        The graph represents all nodes and operations of setup, predict, and update. It also contains prunning() and fingerprint() methods.
    
    mutateProb : float
        The probability to mutate each setup/predict/update operations and operands.
    
    
    Methods
    -------
    
    fingerprint()
        Method used to return fingerPrint of the graph of Alpha
    
    prunning()
        Method used to prun redundant nodes of Alpha.graph that do not contribute to calculate s1
    
    _updateAddOperandProb()
        Method used to calculate the probability to add operands when mutating this Alpha.
    
    mutate_setup()
        Method used to mutate the graph (add/remove/changes Operands, setup opetations) of this Alpha.
    
    mutate_predict()
        Method used to mutate the graph (add/remove/changes Operands, predict opetations) of this Alpha.
    
    mutate_update()
        Method used to mutate the graph (add/remove/changes Operands, update opetations) of this Alpha.
    
    checkS1ConnectsM0_Predict()
        Method used to check whether s1 connects to m0 or s0 or not (in predict operations only)
        -> The purpose is to make sure that s1 connects to m0 or s0 in predict operations after mutating the Alpha
    
    fillUndefinedOperands()
        Method used to add operations to setup operations.
        After mutating, not all operands/nodes are calculated as an output of any operation. -> add an setup operation to initate the value of those undefinedOperands.
    
    mutate()
        Method used to mutate the Graph attribute (Operands/Operations) to generate a new form of graph.
    '''
    
    def __init__(self, graph: Graph = None, mutateProb: float = 0.9, rf: float = 0.02):
        '''
        Initiate an Alpha given a Graph or create a new Graph for Alpha.

        Parameters
        ----------
        graph : Graph, optional
            The graph object defined in 'Graph.py'. The default is None.
        mutateProb : float, optional
            The probability to mutate setup/predict/update in each evolve step.. The default is 0.9.
        rf : float, optional
            risk-free rate. We initiate Scalar s1 by the value of rf. The default is 0.02.

        Returns
        -------
        None.

        '''
        if graph is not None:
            self.graph = copy.deepcopy(graph)
        else:
            self.graph = Graph(rf = rf)
            
        self.mutateProb = mutateProb
    
    def fingerprint(self):
        '''
        Method used to return fingerPrint of the graph of Alpha

        Returns
        -------
        str
            Return the attribute fingerPrint of Alpha.graph

        '''
        return self.graph.fingerprint()
    
    def prunning(self):
        '''
        Method used to prun redundant nodes of Alpha.graph that do not contribute to calculate s1

        Returns
        -------
        None.

        '''
        self.graph.prunning()
    
    def _updateAddOperandProb(self):
        '''
        Method to calculate the probability to add operands when mutating this Alpha.
        The probability calculated is used in mutating steps in mutare_setup(), mutate_predict(), and mutate_update() methods.

        Returns
        -------
        float
            (max number of operands - current number of operands)/max number of operands.
            The more number of operands, the less probability to add more operands to the Alpha.graph.node

        '''
        return min(max((self.graph.maxNumNodes - len(self.graph.nodes))/self.graph.maxNumNodes, 0.0001), 0.9999)
    
    def mutate_setup(self):
        '''
        This method executes a mutation in the setup step.
        
        Steps
        -----
        
            1. Using binorminal distribution with probability of attribute mutateProb to see whether mutate setup or not.
            2. Define p used in step 3: Running method _updateAddOperandProb() to get the probability of adding new nodes.
            3. Mutation:
            3.1. Add Operands and Operations:
                Condition: If binorm(p) is True or too few operations in setupOPs and the number of nodes in attribute graph does not excess the limit (0 and attribute maxNumNodes)
                
                Add Operands: Add at least 1 Scalar/Vector/Matrix and at most 1 Scalar, 1 Vector, 1 Matrix
                Add Operations for each new operand:
                    Scalar: using operation 59 (uniform distribution), and 62 (normal distribution)
                    Vector: using operation 60 (uniform distribution), and 63 (normal distribution)
                            Length of vector is randomized from 2 to 20
                    Matrix: using operation 61 (uniform distribution), and 64 (normal distribution)
                            Shape (i, j) if randomized from (2, 2) to (20, 20)
                    Index: randomized the position we will add to setupOPs list
                    
                    -> run method addSetupOPs of class Graph to add Operation.
            
            3.2. Removing:
                Condition: If binorm(1-p) is True or the number of nodes excess the limit and there is at least 1 operation in setupOPs.
                
                Remove Operations: Randomly pick 1 setup operation. run method removeSetupOPs of class Graph.
                Remove Oerands: run method removeNodes of class Graph to delete.
                -> Make sure we do not delete m0 or s1 in this case.
            
            3.3. Changing Operations:
                Condition: if there is at least 1 operation in setupOPs
                
                Changing: randomly pick 1 operation with output 'key'
                    Scalar: if the value is None -> choose operation 59 or 62
                            if the value is not None -> choose operation 56, 59, or 62
                    Vector: if the value is None -> choose opearation 60 or 63
                            if the value is not None -> choose operation 57, 60, or 63
                    Matrix: if the value is None -> choose operation 61 or 64
                            if the value is not None -> choose operation 58, 61, or 64
                            
        ***NOTE***: 20 is a random number and can be changed but the range need to cover the number of input features/windows
        
        Returns
        -------
        None.

        '''
        if np.random.binomial(1, self.mutateProb): #90% mutate setup
            prob = self._updateAddOperandProb()
            if (np.random.binomial(1, prob) or len(self.graph.setupOPs) <= 1) and 0 <= len(self.graph.nodes) < self.graph.maxNumNodes: #prob% mutating setup by adding Operands
                newNodes = np.random.choice([Scalar(), Vector(), Matrix()], size = np.random.randint(1,4), replace = False)
                for new in newNodes:
                    key = self.graph.addNodes(new)
                    if 's' in key:
                        if np.random.randint(2): #normal(0,1)
                            op = [key, 62, [0, 1]]
                        else: #uniform [-1,1]
                            op = [key, 59, [-1, 1]]
                    
                    if 'v' in key:
                        length = np.random.randint(2, 20)
                        if np.random.randint(2): #norm(0,1)
                            op = [key, 63, [0, 1, length]]
                        else: #uniform[-1,1]
                            op = [key, 60, [-1, 1, length]]
                    
                    if 'm' in key:
                        i = np.random.randint(2, 20)
                        j = np.random.randint(2, 20)
                        if np.random.randint(2): #norm(0,1)
                            op = [key, 64, [0, 1, i, j]]
                        else: #uniform[-1,1]
                            op = [key, 61, [-1, 1, i, j]]
                    
                    index = np.random.randint(len(self.graph.setupOPs)+1)
                    self.graph.addSetupOPs(index, op[0], op[1], op[2])
                
            elif np.random.binomial(1, 1-prob) or len(self.graph.nodes) >= self.graph.maxNumNodes: #(1-prob)% mutating setup by removing Operands
                if len(self.graph.setupOPs) >= 1:
                    key = 'm0'
                    while key in {'m0', 's1'}: #avoid delete m0 and s1
                        op_index = np.random.randint(len(self.graph.setupOPs))
                        key = self.graph.setupOPs[op_index][0]
                    self.graph.removeNodes(key)
                    self.graph.removeSetupOPs(op_index)
                
            elif len(self.graph.setupOPs)>=1: # mutating setup by change operation
                op_index = np.random.randint(len(self.graph.setupOPs))
                op = self.graph.setupOPs[op_index]
                if 's' in op[0]: #scalar
                    if self.graph.nodes[op[0]].value is None:
                        if np.random.randint(2): #normal(0,1)
                            op[1] = 62
                            op[2] = [0, 1]
                        else: #uniform [-1,1]
                            op[1] = 59
                            op[2] = [-1, 1]
                    else:
                        choice = np.random.randint(3)
                        if choice == 2: #constant
                            op[1] = 56
                            op[2] = [float(self.graph.nodes[op[0]].value)]
                        elif choice == 1: #normal(0,1)
                            op[1] = 62
                            op[2] = [0, 1]
                        elif choice == 0: #uniform [-1,1]
                            op[1] = 59
                            op[2] = [-1, 1]
                
                if 'v' in op[0]: #vector
                    if self.graph.nodes[op[0]].value is None:
                        length = np.random.randint(2, 20)
                        if np.random.randint(2): #norm(0,1)
                            op[1] = 63
                            op[2] = [0, 1, length]
                        else: #uniform[-1,1]
                            op[1] = 60
                            op[2] = [-1, 1, length]
                    else:
                        val = float(self.graph.nodes[op[0]].value[np.random.randint(self.graph.nodes[op[0]].shape)])
                        choice = np.random.randint(3)
                        if choice == 2: #constant
                            op[1] = 57
                            op[2] = [val, self.graph.nodes[op[0]].shape]
                        elif choice == 1: # normal (0,1)
                            op[1] = 63
                            op[2] = [0, 1, self.graph.nodes[op[0]].shape]
                        elif choice == 0: #uniform [-1, 1]
                            op[1] = 60
                            op[2] = [-1, 1, self.graph.nodes[op[0]].shape]
                
                if 'm' in op[0]: #matrix
                    if self.graph.nodes[op[0]].value is None:
                        i = np.random.randint(2, 20)
                        j = np.random.randint(2, 20)
                        if np.random.randint(2): #norm(0,1)
                            op[1] = 64
                            op[2] = [0, 1, i, j]
                        else: #uniform[-1,1]
                            op[1] = 61
                            op[2] = [-1, 1, i, j]
                    else:
                        i = np.random.randint(self.graph.nodes[op[0]].shape[0])
                        j = np.random.randint(self.graph.nodes[op[0]].shape[1])
                        val = float(self.graph.nodes[op[0]].value[i, j])
                        i, j = self.graph.nodes[op[0]].shape
                        choice = np.random.randint(3)
                        if choice == 2: #constant
                            op[1] = 58
                            op[2] = [val, i, j]
                        elif choice == 1: # normal (0,1)
                            op[1] = 64
                            op[2] = [0, 1, i, j]
                        elif choice == 0: #uniform [-1, 1]
                            op[1] = 61
                            op[2] = [-1, 1, i, j]
    
    def mutate_predict(self):
        '''
        This method executes a mutation in the predict step.
        
        Steps
        -----
        
            1. Using binorminal distribution with probability of attribute mutateProb to see whether mutate predict or not.
            2. Define p used in step 3: Running method _updateAddOperandProb() to get the probability of adding new nodes.
            3. Create toScalarOp/toVectorOp/toMatrixOp list: list of all operation that output scalar/vector/matrix
                toScalarOp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,65,66,67]
                toVectorOp = [16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53]
                toMatrixOp = [17,28,29,30,32,33,37,38,39,40,41,42,43,46,49]
            4. Mutation:
            4.1. Add Operands and Operations:
                Condition: If binorm(p) is True or too few operations in predictOPs and the number of nodes in attribute graph does not excess the limit (0 and attribute maxNumNodes)
                
                Add Operands: Add at least 1 Scalar/Vector/Matrix and at most 1 Scalar, 1 Vector, 1 Matrix
                Add 2 Operations for each new operand: (look at NOTES below)
                4.1.1: Operation has node as output
                    Create ScalarList/VectorList/MatrixList - list of input nodes (look at NOTES below)
                    
                    Scalar: using operation in toScalarOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    Vector: using operation in toVectorOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    Matrix: using operation in toMatrixOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    
                    -> Then, check the operation is valid or not. if not, skip this mutation
                    else:
                    Index: randomized the position we will add to predictOPs list
                    
                    -> run method addPredictOPs of class Graph to add Operation.
                
                4.1.2: Operation has node as input and scalar as output
                    Choose random output in ScalarList
                    
                    Scalar: using operation [5,6,7,8,9,10,11,12,13,14,15,65,66,67] (scalar -> scalar)
                    Vector: using operation [21,50,54] (vector -> scalar)
                    Matrix: [34,51,55] (matrix -> scalar)
                    
                    -> Then, check the operation is valid or not. if not, skip this mutation
                    else:
                    Index: randomized the position we will add to predictOPs list
                    
                    -> run method addPredictOPs of class Graph to add Operation.
            
            4.2. Removing: (look at NOTES below)
                Condition: If binorm(1-p) is True or the number of nodes excess the limit.
                
                4.2.1: Remove 1 operation only
                    Condition: if binorm(p) is True
                    Remove Operations: Randomly pick 1 predict operation. run method removePredictOPs of class Graph.
                
                4.2.2: Remove 1 operand and all its operations (input or output)
                    Remove Operands: run method removeNodes of class Graph to delete.
                    Remove Operations: for all operations in predictOPs, check whether the deleted node is inpuut or output. if True -> remove operation.
            
            4.3. Changing/Adding Operations without adding Operands:
                Condition: the other cases.
                
                4.3.1: Changing
                    Condition: if binorm(len(predictOPs)/len(nodes)) is true (look at NOTES below)
                4.3.2: Adding
                    Condition: not changing
                
                Both 4.3.1 and 4.3.2 are similar to 4.1.1
                            
        ***NOTES***: 
            + 20 is a random number and can be changed but the range need to cover the number of input features/windows
            + In 4.1, add 2 operations for each node: 1 operation that has node as output.
                    but to avoid prunning this new node and operation, we add 1 more operation that has this node as input
                    and output a scalar (not vector or matrix) to faciliate contribution to predict scalar s1 and avoid requirement of shape of vector or matrix.
            + In 4.1.1, when creating ScalarList/VectorList/MatrixList, we make sure that 1 operation cannot have 1 node as both input and output.
            + In 4.2, make sure we do not delete s1 or m0
            + In 4.3.1 and 4.3.2, the probability is len(predictOPs)/len(nodes)
                -> the more operations in predictOPs -> the more likely to change operation only
                -> the less operations is predictOPs -> the more likely to add operation
        Returns
        -------
        None.

        '''
        if np.random.binomial(1, self.mutateProb): #90% mutate predict
            prob = self._updateAddOperandProb()
            toScalarOp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,65,66,67]
            toVectorOp = [16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53]
            toMatrixOp = [17,28,29,30,32,33,37,38,39,40,41,42,43,46,49]
            
            #prob% mutating predict by adding Operands
            #add operands
            if (np.random.binomial(1, prob) or len(self.graph.predictOPs) <= 1) and 0 <= len(self.graph.nodes) < self.graph.maxNumNodes:
                newNodes = np.random.choice([Scalar(), Vector(), Matrix()], size = np.random.randint(1,4), replace = False)
                
                #Add Operations have 'key' as Output
                for new in newNodes:
                    key = self.graph.addNodes(new)
                    ScalarList = [node for node in self.graph.nodes.keys() if 's' in node and node != key]
                    VectorList = [node for node in self.graph.nodes.keys() if 'v' in node and node != key]
                    MatrixList = [node for node in self.graph.nodes.keys() if 'm' in node and node != key]

                    if 's' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<1000: #loops until selecting a valid operation
                            op = toScalarOp[np.random.randint(len(toScalarOp))]
                            if op in [1,2,3,4,44,47]: #scalar + scalar -> scalar
                                node1, node2 = np.random.choice(ScalarList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op in [5,6,7,8,9,10,11,12,13,14,15,65,66,67]: #scalar -> scalar
                                node = np.random.choice(ScalarList, size = 1)
                                operation = [key, op, [node]]
                            if op in [21,50,54] and len(VectorList) >= 1: # vector -> scalar
                                node = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node]]
                            if op == 27 and len(VectorList) >= 2: #vector + vector -> scalar
                                node1, node2 = np.random.choice(VectorList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op in [34,51,55]: #matrix -> scalar
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                                
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                    
                    if 'v' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<100: #loops until selecting a valid operation
                            op = toVectorOp[np.random.randint(len(toVectorOp))]
                            if op in [16,20,22] and len(VectorList)>=1: #vector -> vector
                                node = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node]]
                            if op == 18 and len(VectorList)>=1: #scalar + vector -> vector
                                node1 = np.random.choice(ScalarList, size = 1)
                                node2 = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op == 19: #scalar + int -> vector
                                node = np.random.choice(ScalarList, size = 1)
                                i = np.random.randint(2,20)
                                operation = [key, op, [node, i]]
                            if op in [23,24,25,26,45,48] and len(VectorList)>=2: # vector + vector -> vector
                                node1, node2 = np.random.choice(VectorList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op == 31 and len(VectorList)>=1: #matrix + vector -> vector
                                node1 = np.random.choice(MatrixList, size = 1)
                                node2 = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op in [35,36,52,53]: #matrix -> vector
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                                
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                        
                    if 'm' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<100: #loops until selecting a valid operation
                            op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                            if op in [17, 30, 37, 38]: #matrix -> matrix
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                            if op == 28 and len(VectorList)>=2: #vector + vector -> matrix
                                node1, node2 = np.random.choice(VectorList, size = 2, replace = False)
                                operation = [key, op, [node1, node2]]
                            if op == 29: #scalar + matrix -> matrix
                                node1 = np.random.choice(ScalarList, size = 1)
                                node2 = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op in [32, 33] and len(VectorList)>=1: #vector + int -> matrix
                                node = np.random.choice(VectorList, size = 1)
                                i = np.random.randint(2,20)
                                operation = [key, op, [node, i]]
                            if op in [39, 40, 41, 42, 43, 46, 49] and len(MatrixList)>=2: #matrix + matrix -> matrix
                                node1, node2 = np.random.choice(MatrixList, size = 2, replace = False)
                                operation = [key, op, [node1, node2]]
                            
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                    
                    index = np.random.randint(len(self.graph.predictOPs)+1)
                    self.graph.addPredictOPs(index, operation[0], operation[1], operation[2])
                    
                    ###############################################################
                    #Add Operations have 'key' as Input and Scalar as output
                    out = np.random.choice(ScalarList)
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        if 's' in key:
                            op = np.random.choice([5,6,7,8,9,10,11,12,13,14,15,65,66,67]) #scalar -> scalar
                        if 'v' in key:
                            op = np.random.choice([21,50,54]) # vector -> scalar
                        if 'm' in key:
                            op = np.random.choice([34,51,55]) # matrix -> scalar
                        operation = [out, op, [key]]
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                    if not valid: 
                        continue
                    else:
                        index = np.random.randint(len(self.graph.predictOPs)+1)
                        self.graph.addPredictOPs(index, operation[0], operation[1], operation[2])
                    
                    
            elif np.random.binomial(1, 1-prob) or len(self.graph.nodes) >= self.graph.maxNumNodes: #(1-prob)% mutating predict by removing Operands
                if np.random.binomial(1, prob): #remove 1 operation only
                    if len(self.graph.predictOPs) >= 1:
                        key = 's1'
                        while key in {'s1', 'm0'}: #force sellecting operation output different from s1, m0
                            op_index = np.random.randint(len(self.graph.predictOPs))
                            key = self.graph.predictOPs[op_index][0]
                        self.graph.removePredictOPs(op_index)
                else: #remove 1 node and its operations
                    key = 's1'
                    while key in {'s1', 'm0'}: #force sellecting operation output different from s1, m0
                        key = np.random.choice(list(self.graph.nodes.keys()))
                    self.graph.removeNodes(key)
                    for op in self.graph.predictOPs:
                        if key == op[0] or key in op[2]:
                            self.graph.predictOPs.remove(op)
                
            else: # mutating predict by change operation
                if np.random.binomial(1, min(0.99,len(self.graph.predictOPs)/len(self.graph.nodes))): #the more predict operation, the more likely to change operation only
                    op_index = np.random.randint(len(self.graph.predictOPs))
                    operation = self.graph.predictOPs[op_index]
                    key, op = operation[0], operation[1]
                    mode = 'change operation'
                else: #the less predict operation, the less likely to change operation, the more likely to add operation
                    op_index = np.random.randint(len(self.graph.predictOPs)+1)
                    key = np.random.choice(list(self.graph.nodes.keys()))
                    if 's' in key: op = toScalarOp[np.random.randint(len(toScalarOp))]
                    if 'v' in key: op = toVectorOp[np.random.randint(len(toVectorOp))]
                    if 'm' in key: op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                    mode = 'add operation'
                
                ScalarList = [node for node in self.graph.nodes.keys() if 's' in node and node != key]
                VectorList = [node for node in self.graph.nodes.keys() if 'v' in node and node != key]
                MatrixList = [node for node in self.graph.nodes.keys() if 'm' in node and node != key]
                
                if 's' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        op = toScalarOp[np.random.randint(len(toScalarOp))]
                        if op in [1,2,3,4,44,47]: #scalar + scalar -> scalar
                            node1, node2 = np.random.choice(ScalarList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op in [5,6,7,8,9,10,11,12,13,14,15,65,66,67]: #scalar -> scalar
                            node = np.random.choice(ScalarList, size = 1)
                            operation = [key, op, [node]]
                        if op in [21,50,54] and len(VectorList) >= 1: # vector -> scalar
                            node = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node]]
                        if op == 27 and len(VectorList) >= 2: #vector + vector -> scalar
                            node1, node2 = np.random.choice(VectorList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op in [34,51,55]: #matrix -> scalar
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                
                if 'v' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        op = toVectorOp[np.random.randint(len(toVectorOp))]
                        if op in [16,20,22] and len(VectorList)>=1: #vector -> vector
                            node = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node]]
                        if op == 18 and len(VectorList)>=1: #scalar + vector -> vector
                            node1 = np.random.choice(ScalarList, size = 1)
                            node2 = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op == 19: #scalar + int -> vector
                            node = np.random.choice(ScalarList, size = 1)
                            i = np.random.randint(2,20)
                            operation = [key, op, [node, i]]
                        if op in [23,24,25,26,45,48] and len(VectorList)>=2: # vector + vector -> vector
                            node1, node2 = np.random.choice(VectorList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op == 31 and len(VectorList)>=1: #matrix + vector -> vector
                            node1 = np.random.choice(MatrixList, size = 1)
                            node2 = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op in [35,36,52,53]: #matrix -> vector
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                    
                if 'm' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<1000: #loops until selecting a valid operation
                        op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                        if op in [17, 30, 37, 38]: #matrix -> matrix
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                        if op == 28 and len(VectorList)>=2: #vector + vector -> matrix
                            node1, node2 = np.random.choice(VectorList, size = 2, replace = False)
                            operation = [key, op, [node1, node2]]
                        if op == 29: #scalar + matrix -> matrix
                            node1 = np.random.choice(ScalarList, size = 1)
                            node2 = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op in [32, 33] and len(VectorList)>=1: #vector + int -> matrix
                            node = np.random.choice(VectorList, size = 1)
                            i = np.random.randint(2,20)
                            operation = [key, op, [node, i]]
                        if op in [39, 40, 41, 42, 43, 46, 49] and len(MatrixList)>=2: #matrix + matrix -> matrix
                            node1, node2 = np.random.choice(MatrixList, size = 2, replace = False)
                            operation = [key, op, [node1, node2]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                if not valid:
                    pass
                else:
                    if mode == 'change operation':
                        self.graph.predictOPs[op_index] = operation
                    elif mode == 'add operation':
                        self.graph.addPredictOPs(op_index, operation[0], operation[1], operation[2])
                    
    def mutate_update(self):
        '''
        This method executes a mutation in the update step.
        
        Steps
        -----
        
            1. Using binorminal distribution with probability of attribute mutateProb to see whether mutate update or not.
            2. Define p used in step 3: Running method _updateAddOperandProb() to get the probability of adding new nodes.
            3. Create toScalarOp/toVectorOp/toMatrixOp list: list of all operation that output scalar/vector/matrix
                toScalarOp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,65,66,67]
                toVectorOp = [16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53]
                toMatrixOp = [17,28,29,30,32,33,37,38,39,40,41,42,43,46,49]
            4. Mutation:
            4.1. Add Operands and Operations:
                Condition: If binorm(p) is True or too few operations in updateOPs and the number of nodes in attribute graph does not excess the limit (0 and attribute maxNumNodes)
                
                Add Operands: Add at least 1 Scalar/Vector/Matrix and at most 1 Scalar, 1 Vector, 1 Matrix
                Add 2 Operations for each new operand: (look at NOTES below)
                4.1.1: Operation has node as output
                    Create ScalarList/VectorList/MatrixList - list of input nodes (look at NOTES below)
                    
                    Scalar: using operation in toScalarOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    Vector: using operation in toVectorOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    Matrix: using operation in toMatrixOp, for each operation, pick inputs from ScalarList/VectorList/MatrixList
                    
                    -> Then, check the operation is valid or not. if not, skip this mutation
                    else:
                    Index: randomized the position we will add to updateOPs list
                    
                    -> run method addUpdateOPs of class Graph to add Operation.
                
                4.1.2: Operation has node as input and scalar as output
                    Choose random output in ScalarList
                    
                    Scalar: using operation [5,6,7,8,9,10,11,12,13,14,15,65,66,67] (scalar -> scalar)
                    Vector: using operation [21,50,54] (vector -> scalar)
                    Matrix: [34,51,55] (matrix -> scalar)
                    
                    -> Then, check the operation is valid or not. if not, skip this mutation
                    else:
                    Index: randomized the position we will add to updateOPs list
                    
                    -> run method addUpdateOPs of class Graph to add Operation.
            
            4.2. Removing: (look at NOTES below)
                Condition: If binorm(1-p) is True or the number of nodes excess the limit.
                
                4.2.1: Remove 1 operation only
                    Condition: if binorm(p) is True
                    Remove Operations: Randomly pick 1 update operation. run method removeUpdateOPs of class Graph.
                
                4.2.2: Remove 1 operand and all its operations (input or output)
                    Remove Operands: run method removeNodes of class Graph to delete.
                    Remove Operations: for all operations in updateOPs, check whether the deleted node is inpuut or output. if True -> remove operation.
            
            4.3. Changing/Adding Operations without adding Operands:
                Condition: the other cases.
                
                4.3.1: Changing
                    Condition: if binorm(len(updateOPs)/len(nodes)) is true (look at NOTES below)
                4.3.2: Adding
                    Condition: not changing
                
                Both 4.3.1 and 4.3.2 are similar to 4.1.1
                            
        ***NOTES***: mutate update is the same as mutate predict
            + 20 is a random number and can be changed but the range need to cover the number of input features/windows
            + In 4.1, add 2 operations for each node: 1 operation that has node as output.
                    but to avoid prunning this new node and operation, we add 1 more operation that has this node as input
                    and output a scalar (not vector or matrix) to faciliate contribution to update scalar s1 and avoid requirement of shape of vector or matrix.
            + In 4.1.1, when creating ScalarList/VectorList/MatrixList, we make sure that 1 operation cannot have 1 node as both input and output.
            + In 4.2, make sure we do not delete s1 or m0
            + In 4.3.1 and 4.3.2, the probability is len(updateOPs)/len(nodes)
                -> the more operations in updateOPs -> the more likely to change operation only
                -> the less operations is updateOPs -> the more likely to add operation
        Returns
        -------
        None.

        '''
        if np.random.binomial(1, self.mutateProb): #90% mutate update
            prob = self._updateAddOperandProb()
            toScalarOp = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,21,27,34,44,47,50,51,54,55,65,66,67]
            toVectorOp = [16,18,19,20,22,23,24,25,26,31,35,36,45,48,52,53]
            toMatrixOp = [17,28,29,30,32,33,37,38,39,40,41,42,43,46,49]
            
            #prob% mutating update by adding Operands
            #add operands
            if (np.random.binomial(1, prob) or len(self.graph.updateOPs) <= 1) and 0 <= len(self.graph.nodes) < self.graph.maxNumNodes:
                newNodes = np.random.choice([Scalar(), Vector(), Matrix()], size = np.random.randint(1,4), replace = False)
                for new in newNodes:
                    key = self.graph.addNodes(new)
                    ScalarList = [node for node in self.graph.nodes.keys() if 's' in node and node != key]
                    VectorList = [node for node in self.graph.nodes.keys() if 'v' in node and node != key]
                    MatrixList = [node for node in self.graph.nodes.keys() if 'm' in node and node != key]
                    
                    #Add Operations have 'key' as Output
                    if 's' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<1000: #loops until selecting a valid operation
                            op = toScalarOp[np.random.randint(len(toScalarOp))]
                            if op in [1,2,3,4,44,47]: #scalar + scalar -> scalar
                                node1, node2 = np.random.choice(ScalarList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op in [5,6,7,8,9,10,11,12,13,14,15,65,66,67]: #scalar -> scalar
                                node = np.random.choice(ScalarList, size = 1)
                                operation = [key, op, [node]]
                            if op in [21,50,54] and len(VectorList) >= 1: # vector -> scalar
                                node = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node]]
                            if op == 27 and len(VectorList) >= 2: #vector + vector -> scalar
                                node1, node2 = np.random.choice(VectorList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op in [34,51,55]: #matrix -> scalar
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                                
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                    
                    if 'v' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<100: #loops until selecting a valid operation
                            op = toVectorOp[np.random.randint(len(toVectorOp))]
                            if op in [16,20,22] and len(VectorList)>=1: #vector -> vector
                                node = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node]]
                            if op == 18 and len(VectorList)>=1: #scalar + vector -> vector
                                node1 = np.random.choice(ScalarList, size = 1)
                                node2 = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op == 19: #scalar + int -> vector
                                node = np.random.choice(ScalarList, size = 1)
                                i = np.random.randint(2,20)
                                operation = [key, op, [node, i]]
                            if op in [23,24,25,26,45,48] and len(VectorList)>=2: # vector + vector -> vector
                                node1, node2 = np.random.choice(VectorList, size = 2)
                                operation = [key, op, [node1, node2]]
                            if op == 31 and len(VectorList)>=1: #matrix + vector -> vector
                                node1 = np.random.choice(MatrixList, size = 1)
                                node2 = np.random.choice(VectorList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op in [35,36,52,53]: #matrix -> vector
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                                
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                        
                    if 'm' in key:
                        valid = False
                        count = 0
                        operation = [None, None, None]
                        while not valid and count<100: #loops until selecting a valid operation
                            op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                            if op in [17, 30, 37, 38]: #matrix -> matrix
                                node = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node]]
                            if op == 28 and len(VectorList)>=2: #vector + vector -> matrix
                                node1, node2 = np.random.choice(VectorList, size = 2, replace = False)
                                operation = [key, op, [node1, node2]]
                            if op == 29: #scalar + matrix -> matrix
                                node1 = np.random.choice(ScalarList, size = 1)
                                node2 = np.random.choice(MatrixList, size = 1)
                                operation = [key, op, [node1, node2]]
                            if op in [32, 33] and len(VectorList)>=1: #vector + int -> matrix
                                node = np.random.choice(VectorList, size = 1)
                                i = np.random.randint(2,20)
                                operation = [key, op, [node, i]]
                            if op in [39, 40, 41, 42, 43, 46, 49] and len(MatrixList)>=2: #matrix + matrix -> matrix
                                node1, node2 = np.random.choice(MatrixList, size = 2, replace = False)
                                operation = [key, op, [node1, node2]]
                                
                            valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                            count += 1
                        if not valid: break
                    
                    index = np.random.randint(len(self.graph.updateOPs)+1)
                    self.graph.addUpdateOPs(index, operation[0], operation[1], operation[2])
                    
                    ###############################################################
                    #Add Operations have 'key' as Input and Scalar as output
                    out = np.random.choice(ScalarList)
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        if 's' in key:
                            op = np.random.choice([5,6,7,8,9,10,11,12,13,14,15,65,66,67]) #scalar -> scalar
                        if 'v' in key:
                            op = np.random.choice([21,50,54]) # vector -> scalar
                        if 'm' in key:
                            op = np.random.choice([34,51,55]) # matrix -> scalar
                        operation = [out, op, [key]]
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                    if not valid:
                        continue
                    else:
                        index = np.random.randint(len(self.graph.updateOPs)+1)
                        self.graph.addUpdateOPs(index, operation[0], operation[1], operation[2])
                    
                    
            elif np.random.binomial(1, 1-prob) or len(self.graph.nodes) >= self.graph.maxNumNodes: #(1-prob)% mutating update by removing Operands
                if np.random.binomial(1, prob): #remove 1 operation
                    if len(self.graph.updateOPs) >= 1:
                        key = 's1'
                        while key in {'s1', 'm0'}: #force sellecting operation output different from s1, m0
                            op_index = np.random.randint(len(self.graph.updateOPs))
                            key = self.graph.updateOPs[op_index][0]
                        self.graph.removeUpdateOPs(op_index)
                else: #remove 1 node and its opereration
                    key = 's1'
                    while key in {'s1', 'm0'}: #force sellecting operation output different from s1, m0
                        key = np.random.choice(list(self.graph.nodes.keys()))
                    self.graph.removeNodes(key)
                    for op in self.graph.updateOPs:
                        if key == op[0] or key in op[2]:
                            self.graph.updateOPs.remove(op)
                
            else: # mutating update by change operation
                if np.random.binomial(1, min(0.99,len(self.graph.updateOPs)/len(self.graph.nodes))): #the more update operations, the more likely to change operation only
                    op_index = np.random.randint(len(self.graph.updateOPs))
                    operation = self.graph.updateOPs[op_index]
                    key, op = operation[0], operation[1]
                    mode = 'change operation'
                else: #the less update operations, the less likely to change operation, the more likely to add operation
                    op_index = np.random.randint(len(self.graph.updateOPs)+1)
                    key = np.random.choice(list(self.graph.nodes.keys()))
                    if 's' in key: op = toScalarOp[np.random.randint(len(toScalarOp))]
                    if 'v' in key: op = toVectorOp[np.random.randint(len(toVectorOp))]
                    if 'm' in key: op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                    mode = 'add operation'
                
                ScalarList = [node for node in self.graph.nodes.keys() if 's' in node and node != key]
                VectorList = [node for node in self.graph.nodes.keys() if 'v' in node and node != key]
                MatrixList = [node for node in self.graph.nodes.keys() if 'm' in node and node != key]
                
                if 's' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        op = toScalarOp[np.random.randint(len(toScalarOp))]
                        if op in [1,2,3,4,44,47]: #scalar + scalar -> scalar
                            node1, node2 = np.random.choice(ScalarList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op in [5,6,7,8,9,10,11,12,13,14,15,65,66,67]: #scalar -> scalar
                            node = np.random.choice(ScalarList, size = 1)
                            operation = [key, op, [node]]
                        if op in [21,50,54] and len(VectorList) >= 1: # vector -> scalar
                            node = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node]]
                        if op == 27 and len(VectorList) >= 2: #vector + vector -> scalar
                            node1, node2 = np.random.choice(VectorList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op in [34,51,55]: #matrix -> scalar
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                
                if 'v' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<100: #loops until selecting a valid operation
                        op = toVectorOp[np.random.randint(len(toVectorOp))]
                        if op in [16,20,22] and len(VectorList)>=1: #vector -> vector
                            node = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node]]
                        if op == 18 and len(VectorList)>=1: #scalar + vector -> vector
                            node1 = np.random.choice(ScalarList, size = 1)
                            node2 = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op == 19: #scalar + int -> vector
                            node = np.random.choice(ScalarList, size = 1)
                            i = np.random.randint(2,20)
                            operation = [key, op, [node, i]]
                        if op in [23,24,25,26,45,48] and len(VectorList)>=2: # vector + vector -> vector
                            node1, node2 = np.random.choice(VectorList, size = 2)
                            operation = [key, op, [node1, node2]]
                        if op == 31 and len(VectorList)>=1: #matrix + vector -> vector
                            node1 = np.random.choice(MatrixList, size = 1)
                            node2 = np.random.choice(VectorList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op in [35,36,52,53]: #matrix -> vector
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                    
                if 'm' in key:
                    valid = False
                    count = 0
                    operation = [None, None, None]
                    while not valid and count<1000: #loops until selecting a valid operation
                        op = toMatrixOp[np.random.randint(len(toMatrixOp))]
                        if op in [17, 30, 37, 38]: #matrix -> matrix
                            node = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node]]
                        if op == 28 and len(VectorList)>=2: #vector + vector -> matrix
                            node1, node2 = np.random.choice(VectorList, size = 2, replace = False)
                            operation = [key, op, [node1, node2]]
                        if op == 29: #scalar + matrix -> matrix
                            node1 = np.random.choice(ScalarList, size = 1)
                            node2 = np.random.choice(MatrixList, size = 1)
                            operation = [key, op, [node1, node2]]
                        if op in [32, 33] and len(VectorList)>=1: #vector + int -> matrix
                            node = np.random.choice(VectorList, size = 1)
                            i = np.random.randint(2,20)
                            operation = [key, op, [node, i]]
                        if op in [39, 40, 41, 42, 43, 46, 49] and len(MatrixList)>=2: #matrix + matrix -> matrix
                            node1, node2 = np.random.choice(MatrixList, size = 2, replace = False)
                            operation = [key, op, [node1, node2]]
                            
                        valid = self.graph._checkValidOP(operation[0], operation[1], operation[2])
                        count += 1
                if not valid:
                    pass
                else:
                    if mode == 'change operation':
                        self.graph.updateOPs[op_index] = operation
                    elif mode == 'add operation':
                        self.graph.addUpdateOPs(op_index, operation[0], operation[1], operation[2])
    
    
    def checkS1ConnectsM0_Predict(self):
        '''
        This method checks whether s1 is connected to m0 or s0 in predict step only by:
            running method checkS1ConnectsM0() of class Graph defined in 'Graph.py'
        
        Returns
        -------
        bool
            if True -> s1 is calculated from m0 or s0.
            if False -> s1 is not calculated from m0 or s0.

        '''
        predict = {}
        for op in self.graph.predictOPs.copy():
            if op[0] in predict:
                self.graph.predictOPs.remove(predict[op[0]])
            predict[op[0]] = op
        return self.graph.checkS1ConnectsM0('s1', {}, predict, {}, {'m0': True, 's0': True})
    
    
    def fillUndefinedOperands(self):
        '''
        After mutating, not all operands/nodes are calculated as an output of any operation. 
        -> add an setup operation to initate the value of those undefined Operands.
        This method used to add operations to setup operations.
        The code is similar to mutate_setup() method.
        
        Returns
        -------
        None.

        '''
        defindedNodes = [operation[0] for operation in self.graph.setupOPs + self.graph.predictOPs + self.graph.updateOPs]
        undefinedNodes = [node for node in self.graph.nodes.keys() if node not in defindedNodes and node not in {'m0', 's0'}]
        
        for node in undefinedNodes:
            if 's' in node:
                if self.graph.nodes[node].value is None:
                    if np.random.randint(2): #normal(0,1)
                        op = [node, 62, [0, 1]]
                    else: #uniform [-1,1]
                        op = [node, 59, [-1, 1]]
                else:
                    choice = np.random.randint(3)
                    if choice == 2: #constant
                        op = [node, 56, [float(self.graph.nodes[node].value)]]
                    elif choice == 1: #normal(0,1)
                        op = [node, 62, [0, 1]]
                    elif choice == 0: #uniform [-1,1]
                        op = [node, 59, [-1, 1]]
            
            if 'v' in node:
                if self.graph.nodes[node].value is None:
                    if self.graph.nodes[node].shape is None:
                        length = np.random.randint(2, 20)
                    else:
                        length = self.graph.nodes[node].shape
                    if np.random.randint(2): #norm(0,1)
                        op = [node, 63, [0, 1, length]]
                    else: #uniform[-1,1]
                        op = [node, 60, [-1, 1, length]]
                else:
                    val = float(self.graph.nodes[node].value[np.random.randint(self.graph.nodes[node].shape)])
                    length = self.graph.nodes[node].shape
                    choice = np.random.randint(3)
                    if choice == 2: #constant
                        op = [node, 57, [val, self.graph.nodes[node].shape]]
                    elif choice == 1: # normal (0,1)
                        op = [node, 63, [0, 1, length]]
                    elif choice == 0: #uniform [-1, 1]
                        op = [node, 60, [-1, 1, length]]
            
            if 'm' in node:
                if self.graph.nodes[node].value is None:
                    if self.graph.nodes[node].shape is None:
                        i = np.random.randint(2, 20)
                        j = np.random.randint(2, 20)
                    else:
                        i, j = self.graph.nodes[node].shape
                    if np.random.randint(2): #norm(0,1)
                        op = [node, 64, [0, 1, i, j]]
                    else: #uniform[-1,1]
                        op = [node, 61, [-1, 1, i, j]]
                else:
                    i = np.random.randint(self.graph.nodes[node].shape[0])
                    j = np.random.randint(self.graph.nodes[node].shape[1])
                    val = float(self.graph.nodes[node].value[i, j])
                    i, j = self.graph.nodes[node].shape
                    choice = np.random.randint(3)
                    if choice == 2: #constant
                        op = [node, 58, [val, i, j]]
                    elif choice == 1: # normal (0,1)
                        op = [node, 64, [0, 1, i, j]]
                    elif choice == 0: #uniform [-1, 1]
                        op = [node, 61, [-1, 1, i, j]]
            
            index = np.random.randint(len(self.graph.setupOPs)+1)
            self.graph.addSetupOPs(index, op[0], op[1], op[2])
    
    
    def mutate(self):
        '''
        This method execute mutating setup, predict, and update operations.
        In mutating predict, we loop 100000 times to faciliate s1 is calculated from m0 or s0.
        Lastly, we run method fillUndefinedOperands() to make a complete graph.

        Returns
        -------
        None.

        '''
        self.mutate_setup()
        self.mutate_predict()
        #make sure s1 connects m0
        cnt = 0
        while (not self.checkS1ConnectsM0_Predict()) and cnt < 100000:
            self.mutate_predict()
            cnt += 1
            if cnt >= 100000:
                self.graph.addPredictOPs(len(self.graph.predictOPs), 's1', np.random.choice([34, 51, 55]), ['m0'])
                #self.graph.show()
        self.mutate_update()
        self.fillUndefinedOperands()
        print('mutate done')
    
if __name__ == '__main__':
    a = Graph()
    
    a.addNodes(Scalar())
    
    a.addNodes(Vector(np.array([1,2,3,4,5])))
    
    a.addNodes(Vector())
    
    a.addNodes(Matrix())
    
    a.addSetupOPs(0, 's2', 59, [0, 1])
    
    a.addSetupOPs(1, 's1', 62, [0,1])
    
    a.addSetupOPs(2, 'm1', 64, [0,1,5,3])

    a.addM0(np.array([[1,2,3], [2,3,4], [3,4,5]]))
    
    a.addPredictOPs(0, 'v2', 35, ['m0'])
    
    a.addPredictOPs(1, 'v1', 31, ['m1', 'v2'])
    
    a.addPredictOPs(2, 's2', 21, ['v1'])
    
    a.addPredictOPs(3, 's1', 6, ['s2'])
    
    a.addUpdateOPs(0, 'v1', 18, ['s2', 'v1'])
    
    a.addUpdateOPs(1, 's2', 6, ['s1'])
    
    a.addUpdateOPs(2, 's1', 5, ['s2'])
    
    a.addUpdateOPs(3, 'm1', 33, ['v1', 3])
    
    a.prunning()
    
    a.fingerprint()
    
    a.show()
    
    b = Alpha(a)

    b.mutate()
    b.graph.show()
    b.prunning()
    b.graph.show()