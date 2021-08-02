from Operands import Scalar, Vector, Matrix
import copy
import numpy as np
np.seterr(all="ignore") 

class Error(Exception):
    def __init__(self, m):
        self.message = m
    def __str__(self):
        return self.message

class Graph():
    '''
    A class used to represent graph of operations and operands of each Alpha
        ...
    Attributes
    -----------
    nodes : dict
        A dictionary which has 's0', 's1', ..., 'v1', ..., 'm0', ... as keys 
        and Scalar, Vector, Matrix objects (defined in 'Operands.py') as values
        Ex: {'s1': Scalar, 'm0': Matrix, 'v6': Vector, 'v7': Vector, 'v8': Vector, 'v15': Vector, 'v16': Vector, 'v17': Vector, 's0': Scalar, 'v25': Vector, 's19': Scalar, 'v26': Vector}
    
    sCount : int
        A counter to count the current number of scalar nodes. It is used to create keys 'si' when we add new Scalar nodes to attribute nodes.
    
    vCount : int
        A counter to count the current number of vector nodes. It is used to create keys 'vi' when we add new Vector nodes to attribute nodes.
    
    mCount : int
        A counter to count the current number of matrix nodes. It is used to create keys 'mi' when we add new Matrix nodes to attribute nodes.
    
    maxNumNodes : int
        The maximum number of nodes in attribute nodes.
    
    allowedSetupOPs : [56, 57, 58, 59, 60, 61, 62, 63, 64]
        The allowed operationse executed in the Setup step.
    
    allowedPredictOPs : [1 -> 56] + [60, 61, 65, 66, 67]
        The allowed operationse executed in the Predict step.
    
    allowedUpdateOPs : [1 -> 56] + [60, 61, 65, 66, 67]
        The allowed operationse executed in the Update step.
    
    ---------
    ***NOTE*** we represent 1 operation as a list: [Output: str, OP: int, Inputs: [key/int,float] ] where:
        + Output: str
            An output key in attribute nodes
        + OP : int
            A number represents operation defined in OPs.py
        + Inputs: [input1, (input2), ...]
            A list of inputs key in attribute nodes or integers/floats depends on which OP is.
        
        Ex: ['v25', 60, [-1, 1, 13]], ['s29', 2, ['s26', 's0']], ['s1', 66, ['s33']], ...
    ---------
    
    setupOPs : list : [operation1, operation2, ...]
        The list of operations executed in order in Setup step.
        Ex: [['v25', 60, [-1, 1, 13]], ['v6', 63, [0, 1, 13]], ['m25', 64, [0, 1, 20, 13]], ['v7', 63, [0, 1, 13]]]
    
    
    predictOPs : list : [operation1, operation2, ...]
        The list of operations executed in order in Predict step.
        Ex: [['s33', 66, ['s41']], ['s19', 50, ['v38']], ['s41', 2, ['s30', 's30']], ['s1', 66, ['s33']]] 
        
    
    updateOPs : list : [operation1, operation2, ...]
        The list of operations executed in order in Update step.
        Ex: [['s47', 2, ['s19', 's19']], ['s33', 44, ['s19', 's29']], ['s0', 12, ['s47']], ['s49', 44, ['s33', 's33']], ['v49', 23, ['v6', 'v37']]]
    
    
    fingerPrint : str
        A string represents the unique of graph.
        Ex: 'v2560v663m2564v763v857v2660v3745v32v32v1745v16v7s292s26s0v1545v8v6s3047s19s29v3823v37v17s3366s41s1950v38s412s30s30s166s33s054v32v3225v25v26v1648v15v6s2651m25s3344s19s29s011s52s4944s33s33v4923v6v37s112s49s1954v49s522s41s49'
    
    
    Methods
    -------
    show()
        Method used to print out the attribute nodes, setupOPs, predictOPs, and updateOPs.
        
    fingerprint()
        Method used to create attribute fingerPrint based on attribute nodes, setupOPs, predictOPs, updateOPs
    
    prunning()
        Method used to prun redundant nodes that do not contribute to predict the value of 's1'
        
    getRedundantNode(node: str, setup: dict, predict: dict, update: dict, redundantNodes: dict)
        Method using Dynamic Programming to get redundant Nodes (redundantNodes) that do not contribute to predict 's1'. 
        This function is used inside method prunning().
    
    checkS1ConnectsM0(node: str, setup: dict, predict: dict, update: dict, connection: dict,  parent: str = None, avoidPath: dict = {})
        Method using Dynamic Programming to check whether S1 depends on M0 or not.
        This function is used inside method prunning().
    
    addS0(data: np.ndarray)
        Method used to update/initiate the value of node S0(actual returns) in attribute nodes.
    
    addM0(data: np.ndarray)
        Method used to update/initiate the value of node M0(data fited to the algorithm) in attribute nodes.
    
    addNodes(node)
        node: Scalar/Vector/Matrix
        Method used to add this new node to attribute nodes.
        Increase the value of attribute sCount/vCount/mCount by 1
        and return the 'key' of this new node in dictionary attribute nodes.
        ('key' will be used in 'Alpha.py')
    
    removeNodes(key: str)
        Method used to remove a node using its 'key' in attribute nodes
    
    addSetupOPs(index: int, Output: str, OP:int, Inputs: list)
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute setupOPs.
    
    addPredictOPs(index: int, Output: str, OP:int, Inputs: list)
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute predictOPs.
    
    addUpdateOPs(index: int, Output: str, OP:int, Inputs: list)
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute updateOPs.
        
    removeSetupOPs(index: int)
        Method used to remove an operation from attribute setupOPs by its position 'index'.
    
    removePredictOPs(index: int)
        Method used to remove an operation from attribute predictOPs by its position 'index'.
    
    removeUpdateOPs(index: int)
        Method used to remove an operation from attribute updateOPs by its position 'index'.
    
    _checkValidOP(Output: str, OP: int, Inputs: list)
        Method used to check whether operation [Output, OP, Inputs] is valid to add into setupOPs/predictOPs/updateOPs by checking its:
            1. Checking type (Scalar, Vector, Matrix)
            2. Checking shape (For Vector and Matrix)
            3. For new nodes created (value = None and shape = None), returns True since they are flexible nodes.
            Actual operations and handling None value of these nodes will be presented in "executeOperation" method of class AlphaEvolve in AlphaEvolve.py.
    
    '''
    
    def __init__(self, nodes: dict = None, setupOPs: list = None, predictOPs: list = None, updateOPs: list = None, maxNumNodes: int = 200, allowedSetupOPs: list = [56, 57, 58, 59, 60, 61, 62, 63, 64], allowedPredictOPs: list = list(range(1,56)) + [60, 61, 65, 66, 67], allowedUpdateOPs: list = list(range(1,56)) + [60, 61, 65, 66, 67], rf: float = 0.02):
        '''
        Initiate a new graph object.

        Parameters
        ----------
        nodes : dict, optional
            A dictionary whose keys are 's0', 's1', ..., 'v1', ..., 'm0', ... and values are Scalar/Vector/Matrix objects. The default is None.
            Ex: {'s1': Scalar, 'm0': Matrix, 'v6': Vector, 'v7': Vector, 'v8': Vector, 'v15': Vector, 'v16': Vector, 'v17': Vector, 's0': Scalar, 'v25': Vector, 's19': Scalar, 'v26': Vector}
    
        setupOPs : list, optional
            A list of Operations executed in Setup step. The default is None.
            Ex: [['v25', 60, [-1, 1, 13]], ['v6', 63, [0, 1, 13]], ['m25', 64, [0, 1, 20, 13]], ['v7', 63, [0, 1, 13]]]
    
        predictOPs : list, optional
            A list of Operations executed in predict step. The default is None.
            Ex: [['s33', 66, ['s41']], ['s19', 50, ['v38']], ['s41', 2, ['s30', 's30']], ['s1', 66, ['s33']]] 
        
        updateOPs : list, optional
            A list of Operations executed in update step. The default is None.
            Ex: [['s47', 2, ['s19', 's19']], ['s33', 44, ['s19', 's29']], ['s0', 12, ['s47']], ['s49', 44, ['s33', 's33']], ['v49', 23, ['v6', 'v37']]]
    
        maxNumNodes : int, optional
            The value of maximum numbers of nodes in attribute nodes. The default is 200.
            
        allowedSetupOPs : list, optional
            The list of all allowed operations executed in setup step. The default is [56, 57, 58, 59, 60, 61, 62, 63, 64].
        
        allowedPredictOPs : list, optional
            The list of all allowed operations executed in predict step. The default is [1 -> 56] + [60, 61, 65, 66, 67].
        
        allowedUpdateOPs : TYPE, optional
            The list of all allowed operations executed in update step. The default is [1 -> 56] + [60, 61, 65, 66, 67].
        
        rf : float, optional
            The risk-free rate assigned to S1 initially. The default is 0.02
            
        Returns
        -------
        None.

        '''
        if nodes is not None:
            self.nodes = copy.deepcopy(nodes)
            self.sCount = max([int(key[1:]) for key in self.nodes.keys() if 's' in key]+[1])+1
            self.vCount = max([int(key[1:]) for key in self.nodes.keys() if 'v' in key]+[0])+1
            self.mCount = max([int(key[1:]) for key in self.nodes.keys() if 'm' in key]+[0])+1
            
            if setupOPs is not None:
                self.setupOPs = copy.deepcopy(setupOPs)
            else:
                self.setupOPs = []
            
            if predictOPs is not None:
                self.predictOPs = copy.deepcopy(predictOPs)
            else:
                self.predictOPs = []
            
            if updateOPs is not None:
                self.updateOPs = copy.deepcopy(updateOPs)
            else:
                self.updateOPs = []
            
        else:
            self.nodes = {'s1': Scalar(rf), 's0': Scalar(0) , 'm0': Matrix()}
            self.sCount = 2
            self.vCount = 1
            self.mCount = 1
            
            self.setupOPs = []
            self.predictOPs = []
            self.updateOPs = []
        
        self.maxNumNodes = maxNumNodes
        self.allowedSetupOPs = allowedSetupOPs
        self.allowedPredictOPs = allowedPredictOPs
        self.allowedUpdateOPs = allowedUpdateOPs
    
    def show(self):
        '''
        This function is used to print out the attribute nodes, setupOPs, predictOPs, and updateOPs.

        '''
        print('NODES:', self.nodes)
        print('SETUP:')
        for op in self.setupOPs:
            print(op)
        print('PREDICT:')
        for op in self.predictOPs:
            print(op)
        print('UPDATE:')
        for op in self.updateOPs:
            print(op)
        #print('FINGERPRINT:', self.fingerprint())
        
    def fingerprint(self):
        '''
        This function is used to create attribute fingerPrint based on attribute nodes, setupOPs, predictOPs, updateOPs
        
        Returns
        -------
        attribute fingerPrint
            represent the uniqueness of operations and operands of the graph.

        '''
        setupPrint = ''
        for op in self.setupOPs:
            setupPrint += (op[0] + str(op[1]))
            
        predictPrint = ''
        for op in self.predictOPs:
            predictPrint += (op[0] + str(op[1]))
            if op[1] not in [19, 32, 33, 56, 57, 58, 59, 60, 61, 62, 63, 64]:
                for inp in op[2]:
                    predictPrint += inp
        
        updatePrint = ''
        for op in self.updateOPs:
            updatePrint += (op[0] + str(op[1]))
            if op[1] not in [19, 32, 33, 56, 57, 58, 59, 60, 61, 62, 63, 64]:
                for inp in op[2]:
                    updatePrint += inp
        
        self.fingerPrint = setupPrint + predictPrint + updatePrint
        return self.fingerPrint
        
    def prunning(self):
        '''
        Method used to prun redundant nodes that do not contribute to predict the value of 's1'
        
        Steps
        ------
            1. Remove operations which will be overwitten. -> Create setup, predict, update dictionary.
                Ex: setup = {'v25': ['v25', 60, [-1, 1, 13]], 'v8': ['v8', 57, [1.0, 13]], ...}
                    predict = {'s1': ['s1', 66, ['s33']], 's19': ['s19', 50, ['v38']], ...}
                    update = {'s0': ['s0', 54, ['v32']], ... }
            2. Create connection dictionary as {'m0': True, 's0': True} and run DP method checkS1ConnectsM0
                to check whether 's1' is calculated from 'm0' or 's0' or not.
            3. If False (s1 does not connect to m0 or s0) -> keep all nodes
                If True (s1 does connect to m0 or s0) 
                -> create redundantNodes dictionary and run DP method getRedundantNode to get those nodes do not contribute to calculation of 's1'.
                -> delete those redundant nodes
            4. Delete the operations whose output or input keys are not in attribute nodes.

        Returns
        -------
        None.

        '''
        # delete operations will be overwritten
        setup = {}
        for op in self.setupOPs.copy():
            if op[0] in {'s0', 's1'}:
                self.setupOPs.remove(op)
                continue
            if op[0] in setup:
                self.setupOPs.remove(setup[op[0]])
            setup[op[0]] = op
        
        predict = {}
        for op in self.predictOPs.copy():
            if op[0] in predict:
                self.predictOPs.remove(predict[op[0]])
            if op[0] in setup:
                self.setupOPs.remove(setup[op[0]])
                del setup[op[0]]
            predict[op[0]] = op
        
        update = {}
        for op in self.updateOPs.copy():
            if op[0] in {'s0', 's1'}:
                self.updateOPs.remove(op)
                continue
            if op[0] in setup:
                self.setupOPs.remove(setup[op[0]])
                del setup[op[0]]
            if op[0] in predict:
                self.predictOPs.remove(predict[op[0]])
                del predict[op[0]]
            if op[0] in update:
                self.updateOPs.remove(update[op[0]])
            update[op[0]] = op
        
        redundantNodes = {}
        # Check s1 connects to m0?
        connection = {'m0': True, 's0': True}
        updateValid, connect_s0, connect_s1 = self.check_update_operands_connect_S0_S1(update)
        if not self.checkS1ConnectsM0('s1', setup, predict, update, connection, 's1'):
            # s1 and m0 are not connected or no operands in update connect to both s0,s1 -> do not delete anything to facilitate s1 connects to m0
            for node in self.nodes.keys():
                redundantNodes[node] = False
                
        elif not updateValid:
            self.getRedundantNode('s1', setup, predict, update, redundantNodes)
            updatedOperands = list(update.keys()).copy()
            for node in updatedOperands:
                self.getRedundantNode(node, {}, {}, update, redundantNodes)
                
            for node in self.nodes.keys():
                if node not in redundantNodes:
                    redundantNodes[node] = True
                    
        else: #s1 connects to m0 -> delete redundant nodes
            self.getRedundantNode('s1', setup, predict, update, redundantNodes)
            updatedOperands = [node for node in list(update.keys()).copy() if connect_s0[node] and connect_s1[node] and node in redundantNodes]
            for node in updatedOperands:
                self.getRedundantNode(node, {}, {}, update, redundantNodes)
                
            for node in self.nodes.keys():
                if node not in redundantNodes:
                    redundantNodes[node] = True
                    
        # delete redundant nodes
        for node in redundantNodes.keys():
            if redundantNodes[node] and node not in {'s1', 's0', 'm0'}:
                del self.nodes[node]
                
        # delete redundant operations of redundant nodes
        for op in self.setupOPs.copy():
            out = op[0]
            for inp in op[2]:
                if (isinstance(inp, str) and inp not in self.nodes) or (out not in self.nodes):
                    self.setupOPs.remove(op)
                    break
        
        for op in self.predictOPs.copy():
            out = op[0]
            for inp in op[2]:
                if (isinstance(inp, str) and inp not in self.nodes) or (out not in self.nodes):
                    self.predictOPs.remove(op)
                    break
                    
        for op in self.updateOPs.copy():
            out = op[0]
            for inp in op[2]:
                if (isinstance(inp, str) and inp not in self.nodes) or (out not in self.nodes):
                    self.updateOPs.remove(op)
                    break
                
    def getRedundantNode(self, node: str, setup: dict, predict: dict, update: dict, redundantNodes: dict):
        '''
        Method using Dynamic Programming to get redundant Nodes (the nodes do not contribute to predict 's1'). 
        This function is used inside method prunning().

        Parameters
        ----------
        node : str
            The current node we are focusing on.
        setup : dict
            setup dictionary created in method prunning().
        predict : dict
            predict dictionary created in method prunning().
        update : dict
            update dictionary created in method prunning().
        redundantNodes : dict
            A dictionary shows whether a node is redundant or not. If True -> delete it
    
        Steps
        -----
            1. Check whether the current node is in redundantNodes dictionary or not. If True -> return None
            2. Assign redundantNodes[key] = False -> this node is not redundant
            3. Create a leaves list - a list of all other nodes that contribute to calculate this current node.
            4. If the leaves is empty -> return None.
                Else: run this method with each node in leaves list as current node.
                
        Returns
        -------
        None.

        '''
        if node in redundantNodes:
            return
        redundantNodes[node] = False
        # create leaves list
        leaves = []
        if node in setup:
            for inp in setup[node][2]:
                if isinstance(inp, str):
                    leaves.append(inp)
        if node in predict:
            for inp in predict[node][2]:
                if isinstance(inp, str):
                    leaves.append(inp)
        if node in update:
            for inp in update[node][2]:
                if isinstance(inp, str):
                    leaves.append(inp)
        
        # if leaves is empty -> pass
        if leaves == []:
            return
        else:
            for leaf in leaves:
                self.getRedundantNode(leaf, setup , predict, update, redundantNodes)
            
    def checkS1ConnectsM0(self, node: str, setup: dict, predict: dict, update: dict, connection: dict,  parent: str = None, avoidPath: dict = {}):
        '''
        Method using Dynamic Programming to check whether S1 depends on M0 or not.
        This function is used inside method prunning().

        Parameters
        ----------
        node : str
            The current node we are focusing on..
        setup : dict
            setup dictionary created in method prunning().
        predict : dict
            predict dictionary created in method prunning().
        update : dict
            update dictionary created in method prunning().
        connection : dict
            A dictionary shows whether the current node connects to m0 or s0.
        parent : str, optional
            The key of nodes that is calculated by this current node. The default is None.
        avoidPath : dict, optional
            A dictionary whose keys are keys of nodes and values are list of ancestor nodes. The default is {}.
        
        Steps
        -----
            1. Add an avoid list to ignore loops in the graph. Add the list to avoidPath dictionary
            2. DP: if node is already hashed in connection -> return connection[node]
            3. Create a leaves list. If this leave list is empty -> return False
            4. Run this method for each leaf in leaves list as current node. If any of them return True -> return True
            5. If None of them return True -> return False.
        
        Returns
        -------
        Bool
            If True -> The current node is connected to m0 or s0.
            If False -> The current node does not connect to m0 or s0.

        '''
        # add avoid list to ignore loops when running DP
        if node == parent: 
            avoidPath[node] = [node]
        else:
            avoid = avoidPath[parent].copy()
            avoid.append(node)
            avoidPath[node] = avoid
        
        # Dynamic Programming
        if node in connection:
            return connection[node]
        
        # create leaves list
        leaves = []
        if node in setup:
            for inp in setup[node][2]:
                if isinstance(inp, str) and inp not in avoidPath[node]:
                    leaves.append(inp)
        if node in predict:
            for inp in predict[node][2]:
                if isinstance(inp, str) and inp not in avoidPath[node]:
                    leaves.append(inp)
        if node in update and node != 's1':
            for inp in update[node][2]:
                if isinstance(inp, str) and inp not in avoidPath[node]:
                    leaves.append(inp)
        
        # if leaves is empty -> no child -> False
        if leaves == []:
            connection[node] = False
        
        # check childs -> if any leaf is True -> True
        for leaf in leaves:
            if self.checkS1ConnectsM0(leaf, setup, predict, update, connection, node, avoidPath):
                connection[node] = True
        # if no leaf is True -> False
        if node not in connection:
            connection[node] = False
            
        return connection[node]
    
    def check_update_operands_connect_S0_S1(self, update):
        connect_s0 = {'s0': True}
        connect_s1 = {'s1': True}
        for node in update.keys():
            self.checkS1ConnectsM0(node, {}, {}, update, connect_s0, node)
            self.checkS1ConnectsM0(node, {}, {}, update, connect_s1, node)
        
        for node in update.keys():
            if connect_s0[node] and connect_s1[node]:
                return True, connect_s0, connect_s1
        return False, connect_s0, connect_s1
    
    def addS0(self, data: np.ndarray):
        '''
        Method used to update/initiate the value of node S0(actual returns) in attribute nodes.

        Parameters
        ----------
        data : np.ndarray
            value of s0 - actual returns.

        Returns
        -------
        None.

        '''
        if 's0' not in self.nodes:
            self.nodes['s0'] = Scalar(data)
        else:
            self.nodes['s0'].updateValue(data)
    
    def addM0(self, data: np.ndarray):
        '''
        Method used to update/initiate the value of node M0(data fited to the algorithm) in attribute nodes.

        Parameters
        ----------
        data : np.ndarray
            value of m0 with the shape of (window, features)

        Returns
        -------
        None.

        '''
        if 'm0' not in self.nodes:
            self.nodes['m0'] = Matrix(data)
        else:
            self.nodes['m0'].updateValue(data)
    
    def addNodes(self, node):
        '''
        Method used to add a new node to attribute nodes
        and increase the value of attribute sCount/vCount/mCount by 1.

        Parameters
        ----------
        node: Scalar/Vector/Matrix
            The new node.

        Returns
        -------
        key : str
            Return the 'key' of the new node in dictionary attribute nodes.
            ('key' will be used in 'Alpha.py')

        '''
        key = ''
        if isinstance(node, Scalar):
            key = 's'+str(self.sCount)
            self.nodes[key] = node
            self.sCount += 1
        if isinstance(node, Vector):
            key = 'v'+str(self.vCount)
            self.nodes[key] = node
            self.vCount += 1
        if isinstance(node, Matrix):
            key = 'm'+str(self.mCount)
            self.nodes[key] = node
            self.mCount += 1
        return key
    
    def removeNodes(self, key: str):
        '''
        Method used to remove a node using its 'key' in attribute nodes

        Parameters
        ----------
        key : str
            The key of the node we want to remove.

        Returns
        -------
        None.

        '''
        if key in self.nodes.keys() and key not in {'s1', 'm0'}:
            del self.nodes[key]
    
    def addSetupOPs(self, index: int, Output: str, OP:int, Inputs: list):
        '''
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute setupOPs.

        Parameters
        ----------
        index : int
            Position that we will add the operation into in the setupOPs list.
        Output : str
            Key of Output nodes.
        OP : int
            Operation defined in OPs.py.
        Inputs : list
            List of Inputs.

        Raises
        ------
        Error
            Error is raised when the Operation is invalid in terms of shape and value.

        Returns
        -------
        None.

        '''
        if not self._checkValidOP(Output, OP , Inputs):
            raise Error("Cannot add invalid Operation")
        if OP in self.allowedSetupOPs and index in range(len(self.setupOPs)+1):
            self.setupOPs.insert(index, [Output, OP, Inputs])
        
    def addPredictOPs(self, index: int, Output: str, OP:int, Inputs: list):
        '''
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute predictOPs.

        Parameters
        ----------
        index : int
            Position that we will add the operation into in the predictOPs list.
        Output : str
            Key of Output nodes.
        OP : int
            Operation defined in OPs.py.
        Inputs : list
            List of Inputs.

        Raises
        ------
        Error
            Error is raised when the Operation is invalid in terms of shape and value.

        Returns
        -------
        None.

        '''
        if not self._checkValidOP(Output, OP , Inputs):
            raise Error("Cannot add invalid Operation")
        if OP in self.allowedPredictOPs and index in range(len(self.predictOPs)+1):
            self.predictOPs.insert(index, [Output, OP, Inputs])
        
    def addUpdateOPs(self, index: int, Output: str, OP:int, Inputs: list):
        '''
        Method used to add the operation [Output, OP, Inputs] to position 'index' in the list attribute updateOPs.

        Parameters
        ----------
        index : int
            Position that we will add the operation into in the updateOPs list.
        Output : str
            Key of Output nodes.
        OP : int
            Operation defined in OPs.py.
        Inputs : list
            List of Inputs.

        Raises
        ------
        Error
            Error is raised when the Operation is invalid in terms of shape and value.

        Returns
        -------
        None.

        '''
        if not self._checkValidOP(Output, OP , Inputs):
            raise Error("Cannot add invalid Operation")
        if OP in self.allowedUpdateOPs and index in range(len(self.updateOPs)+1):
            self.updateOPs.insert(index, [Output, OP, Inputs])
    
    def removeSetupOPs(self, index: int):
        '''
        Method used to remove an operation from attribute setupOPs by its position 'index'.

        Parameters
        ----------
        index : int
            Position of the operation in the setupOPs list we want to delete.

        Returns
        -------
        None.

        '''
        if index in range(len(self.setupOPs)):
            del self.setupOPs[index]
    
    def removePredictOPs(self, index: int):
        '''
        Method used to remove an operation from attribute predictOPs by its position 'index'.

        Parameters
        ----------
        index : int
            Position of the operation in the predictOPs list we want to delete.

        Returns
        -------
        None.

        '''
        if index in range(len(self.predictOPs)):
            del self.predictOPs[index]
    
    def removeUpdateOPs(self, index: int):
        '''
        Method used to remove an operation from attribute updateOPs by its position 'index'.

        Parameters
        ----------
        index : int
            Position of the operation in the updateOPs list we want to delete.

        Returns
        -------
        None.

        '''
        if index in range(len(self.updateOPs)):
            del self.updateOPs[index]
    
    def _checkValidOP(self, Output: str, OP: int, Inputs: list):
        '''
        Method used to check whether operation [Output, OP, Inputs] is valid to add into setupOPs/predictOPs/updateOPs by checking its:
            1. Checking type (Scalar, Vector, Matrix)
            2. Checking shape (For Vector and Matrix)
            3. For new nodes created (value = None and shape = None), returns True since they are flexible nodes.
            Actual operations and handling None value of these nodes will be presented in method "executeOperation" of class AlphaEvolve in AlphaEvolve.py.
    
        Parameters
        ----------
        Output : str
            Key of Output nodes.
        OP : int
            Operation defined in OPs.py.
        Inputs : list
            List of Inputs.

        Returns
        -------
        bool
            If False -> the operation is invalid and cannot add it to the lists of operations.
            If True -> the operation is valid in terms of shape and value (!= None only). -> still add to list of operations.
                    Later, during execution, None shapes and values of nodes can be defined 
                    and they are not satisfied shape and value, then we delete it from list of operations.(showed in AlphaEvolve.executeOperation())

        '''
        if OP in [1, 2, 3, 44, 47]:
            # scalar + scalar -> scalar        
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Inputs[1]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP == 4:
            # scalar + scalar(!=0) -> scalar
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Inputs[1]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    if self.nodes[Inputs[1]].value is None: #input is a new node
                        return True
                    elif self.nodes[Inputs[1]].value != 0:
                        return True
            except:
                return False
        
        if OP == 6:
            # scalar(!=0) -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    if self.nodes[Inputs[0]].value is None: #input is a new node
                        return True
                    elif self.nodes[Inputs[0]].value != 0:
                        return True
            except:
                return False
        
        if OP in [5, 7, 8, 9, 12, 13, 15, 65, 66, 67]:
            # scalar -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP in [10, 11]:
            # scalar(in[-1,1]) -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    if self.nodes[Inputs[0]].value is None: #input is a new node
                        return True
                    elif abs(self.nodes[Inputs[0]].value) <= 1:
                        return True
            except:
                return False
        
        if OP == 14:
            # scalar(>0) -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Output], Scalar):
                    if self.nodes[Inputs[0]].value is None: #input is a new node
                        return True
                    elif self.nodes[Inputs[0]].value > 0:
                        return True
            except:
                return False
        
        if OP in [16, 22]:
            # vector -> vector (same shape)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is a new node
                        return True
                    elif self.nodes[Inputs[0]].shape is None: #input is a new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[0]].shape:
                        return True
            except:
                return False
        
        if OP in [17, 38]:
            # matrix -> matrix (same shape)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None: #output is a new node
                        return True
                    elif self.nodes[Inputs[0]].shape is None: #input is a new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[0]].shape:
                        return True
            except:
                return False
        
        if OP == 18:
            # scalar + vector -> vector (same shape)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is a new node
                         return True
                    elif self.nodes[Inputs[1]].shape is None: #input is a new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[1]].shape:
                        return True
            except:
                return False
        
        if OP == 19:
            # scalar + int -> vector length int
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(Inputs[1], int) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is a new node
                        return True
                    elif self.nodes[Output].shape == Inputs[1]:
                        return True
            except:
                return False
        
        if OP == 20:
            # vector (!=0) -> vector (same shape)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is a new node
                        if self.nodes[Inputs[0]].value is None: #input is a new node
                            return True
                        elif (self.nodes[Inputs[0]].value!=0).all():
                            return True
                    elif self.nodes[Inputs[0]].shape is None: #input is a new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[0]].shape and (self.nodes[Inputs[0]].value!=0).all():
                        return True
            except:
                return False
        
        if OP in [21, 50, 54]:
            # vector -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP in [23, 24, 25, 45, 48]:
            # vector + vector -> vector (same shape)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is a new node
                        if self.nodes[Inputs[0]].shape is None or self.nodes[Inputs[1]].shape is None: #input is a new node
                            return True
                        elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape:
                            return True
                    elif self.nodes[Inputs[0]].shape is None: #input 0 is a new node
                        if self.nodes[Inputs[1]].shape is None: #input 1 is a new node
                            return True
                        elif self.nodes[Inputs[1]].shape == self.nodes[Output].shape:
                            return True
                    elif self.nodes[Inputs[1]].shape is None: # input 1 is a new node
                        if self.nodes[Inputs[0]].shape == self.nodes[Output].shape:
                            return True
                    elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape == self.nodes[Output].shape:# no new nodes
                        return True
            except:
                return False
        
        if OP == 26:
            # vector + vector(!=0) -> vector (same shape)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Inputs[1]].shape is None: #input 1 is a new node -> no need to check value != 0
                        if self.nodes[Inputs[0]].shape is None or self.nodes[Output].shape is None: #input is a new node
                            return True
                        elif self.nodes[Inputs[0]].shape == self.nodes[Output].shape:
                            return True
                    elif self.nodes[Inputs[0]].shape is None: #input 0 is a new node
                        if self.nodes[Output].shape is None: #output is a new node
                            return True
                        elif self.nodes[Inputs[1]].shape == self.nodes[Output].shape and (self.nodes[Inputs[1]].value!=0).all():
                            return True
                    elif self.nodes[Output].shape is None: #output is a new node
                        if self.nodes[Inputs[1]].shape == self.nodes[Inputs[0]].shape and (self.nodes[Inputs[1]].value!=0).all():
                            return True
                    elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape == self.nodes[Output].shape and (self.nodes[Inputs[1]].value!=0).all():# no new nodes
                        return True
            except:
                return False
        
        if OP == 27:
            # vector + vector(same shape) -> scalar
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Scalar):
                    if self.nodes[Inputs[0]].shape is None or self.nodes[Inputs[1]].shape is None: #input is a new node
                        return True
                    elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape:
                        return True
            except:
                return False
        
        if OP == 28:
            # vector (length m) + vector (length n) -> matrix (m x n)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None: # output is a new node
                        return True
                    elif (self.nodes[Inputs[0]].shape, self.nodes[Inputs[1]].shape) in [self.nodes[Output].shape, (None, self.nodes[Output].shape[1]), (self.nodes[Output].shape[0], None), (None, None)]:
                        return True
            except:
                return False
        
        if OP == 29:
            # scalar + matrix -> matrix (same shape)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Scalar) and isinstance(self.nodes[Inputs[1]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None or self.nodes[Inputs[1]].shape is None: # input or output matrix is new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[1]].shape: # both matrices are not new node
                        return True
            except:
                return False
        
        if OP == 30:
            # matrix (!=0) -> matrix (same shape)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Inputs[0]].shape is None: #input is new node -> no check value != 0
                        return True
                    elif self.nodes[Output].shape is None and (self.nodes[Inputs[0]].value!=0).all(): #output is new node
                        return True
                    elif self.nodes[Output].shape == self.nodes[Inputs[0]].shape and (self.nodes[Inputs[0]].value!=0).all(): #both are not new nodes
                        return True
            except:
                return False
        
        if OP == 31:
            # matrix (m x n) + vector (n) -> vector (m)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Inputs[1]], Vector) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Inputs[0]].shape is None: # input matrix is new node -> can flexible the rest vector
                        return True
                    elif (self.nodes[Output].shape, self.nodes[Inputs[1]].shape) in [self.nodes[Inputs[0]].shape, (None, self.nodes[Inputs[0]].shape[1]), (self.nodes[Inputs[0]].shape[0], None), (None, None)]:
                        return True
            except:
                return False
        
        if OP == 32:
            # vector (n) + int -> matrix (i x n)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(Inputs[1], int) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None:
                        return True
                    elif (Inputs[1], self.nodes[Inputs[0]].shape) in [self.nodes[Output].shape, (self.nodes[Output].shape[0], None)]:
                        return True
            except:
                return False
        
        if OP == 33:
            # vector (n) + int -> matrix (n x i)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Vector) and isinstance(Inputs[1], int) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None:
                        return True
                    elif (self.nodes[Inputs[0]].shape, Inputs[1]) in [self.nodes[Output].shape, (None, self.nodes[Output].shape[0])]:
                        return True
            except:
                return False
        
        if OP in [34, 51, 55]:
            # matrix -> scalar
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP in [35, 52, 53]:
            # matrix (n x m) -> vector (n)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Vector):
                     if self.nodes[Output].shape is None or self.nodes[Inputs[0]].shape is None:
                         return True
                     elif self.nodes[Inputs[0]].shape[0] == self.nodes[Output].shape:
                         return True
            except:
                return False
        
        if OP == 36:
            # matrix (n x m) -> vector (m)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Vector):
                     if self.nodes[Output].shape is None or self.nodes[Inputs[0]].shape is None:
                         return True
                     elif self.nodes[Inputs[0]].shape[1] == self.nodes[Output].shape:
                         return True
            except:
                return False
                    
        if OP == 37:
            # matrix (n x m) -> matrix (m x n)
            try:
                if len(Inputs) == 1 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None or self.nodes[Inputs[0]].shape is None:
                        return True
                    elif self.nodes[Output].shape == (self.nodes[Inputs[0]].shape[1], self.nodes[Inputs[0]].shape[0]):
                        return True
            except:
                return False
        
        if OP in [39, 40, 41, 46, 49]:
            # matrix + matrix -> matrix (same size)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Inputs[1]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Inputs[1]].shape is None: #input 1 is new node
                        if self.nodes[Inputs[0]].shape is None or self.nodes[Output].shape is None: # input 0 or output is new node
                            return True
                        elif self.nodes[Inputs[0]].shape == self.nodes[Output].shape: #only input 1 is new node
                            return True
                    elif self.nodes[Inputs[0]].shape is None: # input 0 is new node
                        if self.nodes[Output].shape is None: # output is new node
                            return True
                        elif self.nodes[Inputs[1]].shape == self.nodes[Output].shape: # only input 0 is new node
                            return True
                    elif self.nodes[Output].shape is None: #output is new node
                        if self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape:
                            return True
                    elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape == self.nodes[Output].shape: # no new node
                        return True
            except:
                return False
        
        if OP == 42:
            # matrix + matrix(!=0) -> matrix (same size)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Inputs[1]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Inputs[1]].shape is None: #input 1 is new node
                        if self.nodes[Inputs[0]].shape is None or self.nodes[Output].shape is None: # input 0 or output is new node
                            return True
                        elif self.nodes[Inputs[0]].shape == self.nodes[Output].shape: #only input 1 is new node
                            return True
                    elif self.nodes[Inputs[0]].shape is None: # input 0 is new node
                        if self.nodes[Output].shape is None and (self.nodes[Inputs[1]].value!=0).all(): # output is new node
                            return True
                        elif self.nodes[Inputs[1]].shape == self.nodes[Output].shape and (self.nodes[Inputs[1]].value!=0).all(): # only input 0 is new node
                            return True
                    elif self.nodes[Output].shape is None: #output is new node
                        if self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape and (self.nodes[Inputs[1]].value!=0).all():
                            return True
                    elif self.nodes[Inputs[0]].shape == self.nodes[Inputs[1]].shape == self.nodes[Output].shape and (self.nodes[Inputs[1]].value!=0).all(): # no new node
                        return True
            except:
                return False
        
        if OP == 43:
            # matrix (m x n) + matrix (n x p) -> matrix (m x p)
            try:
                if len(Inputs) == 2 and isinstance(self.nodes[Inputs[0]], Matrix) and isinstance(self.nodes[Inputs[1]], Matrix) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None: #output is new node
                        if self.nodes[Inputs[0]].shape is None or self.nodes[Inputs[1]].shape is None: # input is new node
                            return True
                        elif self.nodes[Inputs[0]].shape[1] == self.nodes[Inputs[1]].shape[0]: # only output is new node
                            return True
                    elif self.nodes[Inputs[1]].shape is None: # input 1 is new node
                        if self.nodes[Inputs[0]].shape is None: # input 0 is also new node
                            return True
                        elif self.nodes[Inputs[0]].shape[0] == self.nodes[Output].shape[0]: #only input 1 is new node
                            return True
                    elif self.nodes[Inputs[0]].shape is None: #only input 0 is new node
                        if self.nodes[Inputs[1]].shape[1] == self.nodes[Output].shape[1]:
                            return True
                    elif self.nodes[Inputs[0]].shape[1] == self.nodes[Inputs[1]].shape[0] and self.nodes[Output].shape == (self.nodes[Inputs[0]].shape[0], self.nodes[Inputs[1]].shape[1]): #no new node
                        return True
            except:
                return False
        
        if OP == 56:
            # const -> scalar
            try:
                if len(Inputs) == 1 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP == 57:
            # const + int -> vector (length i)
            try:
                if len(Inputs) == 2 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and isinstance(Inputs[1], int) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is new node
                        return True
                    elif self.nodes[Output].shape == Inputs[1]:
                        return True
            except:
                return False
        
        if OP == 58:
            # const + int + int -> matrix (int x int)
            try:
                if len(Inputs) == 3 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and isinstance(Inputs[1], int) and isinstance(Inputs[2], int) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None: #output is new node
                        return True
                    elif self.nodes[Output].shape == (Inputs[1], Inputs[2]):
                        return True
            except:
                return False
        
        if OP == 59:
            # float + float -> scalar
            try:
                if len(Inputs) == 2 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(self.nodes[Output], Scalar):
                    return True
            except:
                return False
        
        if OP == 60:
            # float + float + int -> vector
            try:
                if len(Inputs) == 3 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(Inputs[2], int) and isinstance(self.nodes[Output], Vector):
                    if self.nodes[Output].shape is None: #output is new node
                        return True
                    elif self.nodes[Output].shape == Inputs[2]:
                        return True
            except:
                return False
        
        if OP == 61:
            # float + float + int + int -> matrix
            try:
                if len(Inputs) == 4 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(Inputs[2], int) and isinstance(Inputs[3], int) and isinstance(self.nodes[Output], Matrix):
                    if self.nodes[Output].shape is None: #output is new node
                        return True
                    elif self.nodes[Output].shape == (Inputs[2], Inputs[3]):
                        return True
            except:
                return False
        
        if OP == 62:
            # float + float(>0) -> scalar
            try:
                if len(Inputs) == 2 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(self.nodes[Output], Scalar) and Inputs[1] > 0:
                    return True
            except:
                return False
        
        if OP == 63:
            # float + float(>0) + int -> vector
            try:
                if len(Inputs) == 3 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(Inputs[2], int) and isinstance(self.nodes[Output], Vector) and Inputs[1] > 0:
                    if self.nodes[Output].shape is None: #output is new node
                        return True
                    elif self.nodes[Output].shape == Inputs[2]:
                        return True
            except:
                return False
        
        if OP == 64:
            # float + float(>0) + int + int -> matrix
            try:
                if len(Inputs) == 4 and (isinstance(Inputs[0], float) or isinstance(Inputs[0], int)) and (isinstance(Inputs[1], float) or isinstance(Inputs[1], int)) and isinstance(Inputs[2], int) and isinstance(Inputs[3], int) and isinstance(self.nodes[Output], Matrix) and Inputs[1] > 0:
                    if self.nodes[Output].shape is None:
                        return True
                    elif self.nodes[Output].shape == (Inputs[2], Inputs[3]):
                        return True
            except:
                return False
        
        return False