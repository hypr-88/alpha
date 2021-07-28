import numpy as np
np.seterr(all="ignore") 

class Error(Exception):
    def __init__(self, m):
        self.message = m
    def __str__(self):
        return self.message
    
class Scalar():
    '''
    A class used to represent scalar operands
        ...
    
    Attributes
    -----------
    value: np.ndarray(dtype = float32)
        The value of scalar operand
    shape: 0
        The shape of scalar (always equal to 0)
        
    
    Methods
    -----------
    updateValue(s)
        Assign the value s to attribute value
    '''
    
    
    def __init__(self, s = None):
        '''
        
        Parameters
        ----------
        s : float/int/np.ndarray, optional
            Create a new scalar object with value of s. The default is None.

        Returns
        -------
        None.
        '''
        
        if s is not None:
            self.updateValue(s)
        else:
            self.value = None
        self.shape = 0
        
    def updateValue(self, s):
        '''
        
        Parameters
        ----------
        s : float/int/np.ndarray with shape of 1
            This function assign s to attribute value. It can handle the all types of scalar input(int, float, np.array with shape of:(), (1,), (1,1), (1,1,1), ...)

        Raises
        ------
        Error
            Raise error when the input is not scalar number.

        Returns
        -------
        None.

        '''
        if isinstance(s, np.ndarray) or isinstance(s, np.int32) or isinstance(s, np.int64) or isinstance(s, np.float32) or isinstance(s, np.float64):
            if len(s.shape) == 0:
                s = float(s)
            elif sum(s.shape)/len(s.shape) == 1:
                s = float(s)
        if isinstance(s, float) or isinstance(s, int):
            s = np.array(s, dtype = np.float32)
            self.value = s
        else:
            self.value = None
            raise Error("Input is not scalar")
    
    def __repr__(self):
        return "Scalar"
    
class Vector():
    '''
    A class used to represent vector operands
        ...
    
    Attributes
    -----------
    value: np.ndarray(dtype = float32)
        The value of vector operand
    shape: int
        The length of vector operands
        
    
    Methods
    -----------
    updateValue(v)
        Assign the value v to attribute value
    '''


    def __init__(self, v = None):
        '''

        Parameters
        ----------
        v : array, optional
            Create a new vector object with value of v. The default is None.

        Returns
        -------
        None.

        '''
        self.shape = None
        if v is not None:
            self.updateValue(v)
        else:
            self.value = None
        
    def updateValue(self, v):
        '''

        Parameters
        ----------
        v : array alike
            This function assign v to attribute value. It can handle types of vector input of list, 1D-np.matrix, np.array with shape of (n,)

        Raises
        ------
        Error
            Raise error when the input is not 1D vector or the input has different shape from the current attribute shape.

        Returns
        -------
        None.

        '''
        if isinstance(v, np.matrix):
            v = np.asarray(v, dtype = np.float32)
        if not isinstance(v, np.ndarray):
            try:
                v = np.array(v, dtype = np.float32)
            except:
                self.value = None
                raise Error("Input is not vector")
        
        if len(v.shape) == 0:
            self.value = None
            raise Error("Input is not vector")
            
        # update vector with the same shape or initialized
        if self.shape == len(v) or self.shape is None:
            self.value = v
            self.shape = len(self.value)
        # handle np.array with shape of (1,n), (1,1,n), (1,1,1,n), ...
        #elif len(v.shape) != 1 and v.shape[-1] != 1 and sum(v.shape[:-1])/len(v.shape[:-1]) == 1:
        #   if self.shape == len(v) or self.shape is None:
        #        self.value = np.squeeze(v)
        #        self.shape = len(self.value)
        else:
            self.value = None
            raise Error("Input is not satisfied")
            
    def __repr__(self):
        return "Vector"
    
class Matrix():
    '''
    A class used to represent matrix operands
        ...
    
    Attributes
    -----------
    value: np.ndarray(dtype = float32)
        The value of matrix operand
    shape: tuple
        The shape of matrix operands
        
    
    Methods
    -----------
    updateValue(m)
        Assign the value m to attribute value
    '''
    def __init__(self, m = None):
        '''
        
        Parameters
        ----------
        m : 2D-Matrix/np.ndarray/np.matrix, optional
            Create a new matrix object with value of m. The default is None.

        Returns
        -------
        None.

        '''
        
        self.shape = None
        if m is not None:
            self.updateValue(m)
        else:
            self.value = None
        
    def updateValue(self, m):
        '''

        Parameters
        ----------
        m : 2D-Matrix/np.ndarray/np.matrix
            This function assign m to attribute value.
            
        Raises
        ------
        Error
            Raise error when the input is not 2D or the input has different shape from the current attribute shape.

        Returns
        -------
        None.

        '''
        
        if isinstance(m, np.matrix):
            m = np.asarray(m, dtype = np.float32)
        if not isinstance(m, np.ndarray):
            try:
                m = np.array(m, dtype = np.float32)
            except:
                self.value = None
                raise Error("Input is not matrix")
        if len(m.shape) != 2:
            self.value = None
            raise Error("Input is not matrix")
        else:
            if self.shape is None or self.shape == m.shape:
                self.value = m
                self.shape = self.value.shape
            else:
                raise Error("New matrix must have the same shape")
    
    def __repr__(self):
        return "Matrix"