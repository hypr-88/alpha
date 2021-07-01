import cupy as cp
import pandas as pd
from scipy.stats import rankdata
from Operands import Scalar, Vector, Matrix

'''
Notes: s stands for Scalar, v stands for Vector, m stands for Matrix
Definition of 67 functions of operations:
    1. s = s + s
    2. s = s - s
    3. s = s * s
    4. s = s / s
    5. s = |s|
    6. s = 1 / s
    7. s = sin(s)
    8. s = cos(s)
    9. s = tan(s)
    10. s = arcsin(s)
    11. s = arccos(s)
    12. s = arctan(s)
    13. s = e^s
    14. s = log(s)
    15. s = 1, if s>0
            0, otherwise
    16. v[i] = 1 if v[i] > 0
                0, otherwise
    17. m[i,j] = 1 if m[i,j] > 0
                0, otherwise
    18. v[i] = s * v[i]
    19. v[i] = s, for all i
    20. v[i] = 1 / v[i]
    21. s = norm(v)
    22. v[i] = |v[i]|
    23. v[i] = v[i] + v[i]
    24. v[i] = v[i] - v[i]
    25. v[i] = v[i] * v[i]
    26. v[i] = v[i] / v[i]
    27. s = dot(v, v)
    28. m[i,j] = v[i]*v[j]
    29. m[i,j] = s * m[i,j]
    30. m[i,j] = 1 / m[i,j]
    31. v = dot(m, v)
    32. m[,j] = v[j]
    33. m[i,] = v[i]
    34. s = norm(m)
    35. v[i] = norm(m[i,])
    36. v[j] = norm(m[,j])
    37. m = transpose(m)
    38. m[i,j] = |m[i,j]|
    39. m[i,j] = m[i,j] + m[i,j]
    40. m[i,j] = m[i,j] - m[i,j]
    41. m[i,j] = m[i,j] * m[i,j]
    42. m[i,j] = m[i,j] / m[i,j]
    43. m = matmul(m, m)
    44. s = min(s1, s2)
    45. v[i] = min(v1[i], v2[i])
    46. m[i,j] = min(m1[i,j], m2[i,j])
    47. s = max(s1, s2)
    48. v[i] = max(v1[i], v2[i])
    49. m[i,j] = max(m1[i,j], m2[i,j])
    50. s = mean(v)
    51. s = mean(m)
    52. v[i] = mean(m[i,])
    53. v[i] = std(m[i,])
    54. s = std(v)
    55. s = std(m)
    56. s = const
    57. v[i] = const, for all i
    58. m[i,j] = const, for all i,j
    59. s ~ Uniform(a, b)
    60. v[i] ~ Uniform(a, b)
    61. m[i,j] ~ Uniform(a, b)
    62. s ~ Normal(a, b)
    63. v[i] ~ Normal(a, b)
    64. m[i,j] ~ Normal(a, b)
    65. s = rank(s)
    66. s = rankIndustry(s)
    67. s = s - mean(s_in_same_Industry)
    
'''
class inputError(Exception):
    def __init__(self, m):
        self.message = m
    def __str__(self):
        return self.message

def OP1(s1: Scalar, s2: Scalar) -> cp.float32:
    # s1 + s2
    return (s1.value + s2.value)

def OP2(s1: Scalar, s2: Scalar) -> cp.float32:
    # s1 - s2
    return (s1.value - s2.value)

def OP3(s1: Scalar, s2: Scalar) -> cp.float32:
    # s1 * s2
    return (s1.value*s2.value)

def OP4(s1: Scalar, s2: Scalar) -> cp.float32:
    # s1 / s2
    if s2.value == 0:
        raise(inputError("Divisor cannot be 0"))
    return (s1.value/s2.value)

def OP5(s: Scalar) -> cp.float32:
    # |s|
    return (abs(s.value))

def OP6(s: Scalar) -> cp.float32:
    # 1/s
    if s.value == 0:
        raise(inputError("Cannot inverse 0"))
    return (1/s.value)

def OP7(s: Scalar) -> cp.float32:
    # sin(s)
    return (cp.sin(s.value))

def OP8(s: Scalar) -> cp.float32:
    # cos(s)
    return (cp.cos(s.value))

def OP9(s: Scalar) -> cp.float32:
    # tan(s)
    return (cp.tan(s.value))

def OP10(s: Scalar) -> cp.float32:
    # arcsin(s)
    if abs(s.value) > 1:
        raise(inputError("Input must be in the range [-1,1]"))
    return (cp.arcsin(s.value))

def OP11(s: Scalar) -> cp.float32:
    # arccos(s)
    if abs(s.value) > 1:
        raise(inputError("Input must be in the range [-1,1]"))
    return (cp.arccos(s.value))

def OP12(s: Scalar) -> cp.float32:
    # arctan(s)
    return (cp.arctan(s.value))

def OP13(s: Scalar) -> cp.float32:
    # e^s
    return (cp.exp(s.value))

def OP14(s: Scalar) -> cp.float32:
    # ln(s)
    if s.value <= 0:
        raise(inputError("Input must be positive"))
    return (cp.log(s.value))

def OP15(s: Scalar) -> cp.float32:
    # heaviside(s) for scalar
    return (max(cp.sign(s.value), 0))

def OP16(v: Vector) -> cp.ndarray:
    # heaviside(v) for vector
    return (cp.heaviside(v.value, 0))

def OP17(m: Matrix) -> cp.ndarray:
    # heaviside(m) for matrix
    return (cp.heaviside(m.value, 0))

def OP18(s: Scalar, v: Vector) -> cp.ndarray:
    # s*v
    return (s.value*v.value)

def OP19(s: Scalar, i: int) -> cp.ndarray:
    # v = bcast(s): scalar to vector
    # i: length of vector output
    return cp.array([s.value] * i)

def OP20(v: Vector) -> cp.ndarray:
    # 1/v: inverse of vector
    if (v.value==0).any():
        raise(inputError("Element cannot be 0"))
    return (1/v.value)

def OP21(v: Vector) -> cp.float32:
    # ||v||
    return (cp.linalg.norm(v.value))

def OP22(v: Vector) -> cp.ndarray:
    # |v|
    return (abs(v.value))

def OP23(v1: Vector, v2: Vector) -> cp.ndarray:
    # v1+v2
    if len(v1.value)!=len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value + v2.value)

def OP24(v1: Vector, v2: Vector) -> cp.ndarray:
    # v1-v2
    if len(v1.value)!=len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value - v2.value)

def OP25(v1: Vector, v2: Vector) -> cp.ndarray:
    # v1*v2
    if len(v1.value)!=len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value * v2.value)

def OP26(v1: Vector, v2: Vector) -> cp.ndarray:
    # v1/v2
    if len(v1.value)!=len(v2.value):
        raise inputError("Vectors input must have the same size")
    if (v2.value==0).any():
        raise inputError("Divisor input cannot be 0")
    return (v1.value / v2.value)

def OP27(v1: Vector, v2: Vector) -> cp.float32:
    # dot product of 2 vectors
    if len(v1.value)!=len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (cp.dot(v1.value, v2.value))

def OP28(v1: Vector, v2: Vector) -> cp.ndarray:
    # outer product
    return (cp.outer(v1.value, v2.value))

def OP29(s: Scalar, m: Matrix) -> cp.ndarray:
    # s*m
    return (s.value*m.value)

def OP30(m: Matrix) -> cp.ndarray:
    # 1/m
    if (m.value==0).any():
        raise(inputError("Element cannot be 0"))
    return (1/m.value)

def OP31(m: Matrix, v: Vector) -> cp.ndarray:
    # dot(m, v)
    if m.value.shape[1]!=len(v.value):
        raise inputError("Vector input must have the same size as row size of matrix")
    return cp.dot(m.value, v.value)

def OP32(v: Vector, i: int) -> cp.ndarray:
    # [[1, 2, 3, 4],
    # [1, 2, 3, 4],
    # [1, 2, 3, 4]]
    return cp.array([v.value] * i)

def OP33(v: Vector, j: int) -> cp.ndarray:
    # [[1, 1, 1],
    # [2, 2, 2],
    # [3, 3, 3],
    # [4, 4, 4]]
    return (cp.transpose(cp.array([v.value] * j)))

def OP34(m: Matrix) -> cp.float32:
    # ||m||
    return (cp.linalg.norm(m.value))

def OP35(m: Matrix) -> cp.ndarray:
    # v[i] = norm(m[i,])
    return (cp.linalg.norm(m.value, axis = 1))

def OP36(m: Matrix) -> cp.ndarray:
    # v[j] = norm(m[,j])
    return (cp.linalg.norm(m.value, axis = 0))

def OP37(m: Matrix) -> cp.ndarray:
    # transpose(m)
    return (cp.transpose(m.value))

def OP38(m: Matrix) -> cp.float32:
    # |m|
    return (abs(m.value))

def OP39(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # m1+m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value + m2.value)

def OP40(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # m1-m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value - m2.value)

def OP41(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # m1*m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value * m2.value)

def OP42(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # m1/m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    if (m2.value == 0).any():
        raise(inputError("Element cannot be 0"))
    return (m1.value / m2.value)

def OP43(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # matmul(m1, m2)
    return (cp.matmul(m1.value, m2.value))

def OP44(s1: Scalar, s2: Scalar) -> cp.float32:
    # min(s1, s2)
    return (min(s1.value, s2.value))

def OP45(v1: Vector, v2: Vector) -> cp.ndarray:
    # min(v1, v2)
    if len(v1.value)!= len(v2.value):
        raise inputError("Vectors must have the same size")
    return (cp.minimum(v1.value, v2.value))

def OP46(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # min(m1, m2)
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same shape")
    return (cp.minimum(m1.value, m2.value))

def OP47(s1: Scalar, s2: Scalar) -> cp.float32:
    # max(s1, s2)
    return (max(s1.value, s2.value))

def OP48(v1: Vector, v2: Vector) -> cp.ndarray:
    # max(v1, v2)
    if len(v1.value)!= len(v2.value):
        raise inputError("Vectors must have the same size")
    return (cp.maximum(v1.value, v2.value))

def OP49(m1: Matrix, m2: Matrix) -> cp.ndarray:
    # max(m1, m2)
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same shape")
    return (cp.maximum(m1.value, m2.value))

def OP50(v: Vector) -> cp.float32:
    # mean(v)
    return (cp.mean(v.value))

def OP51(m: Matrix) -> cp.float32:
    # mean(m)
    return (cp.mean(m.value))

def OP52(m: Matrix) -> cp.ndarray:
    # vector mean of each row
    return (cp.mean(m.value, axis = 1))

def OP53(m: Matrix) -> cp.ndarray:
    # vector std of each row
    return (cp.std(m.value, axis = 1))

def OP54(v: Vector) -> cp.float32:
    # std of vector
    return (cp.std(v.value))

def OP55(m: Matrix) -> cp.float32:
    # std of matrix
    return (cp.std(m.value))

def OP56(const: float) -> float:
    # Initiate constant scalar
    return const

def OP57(const: float, i: int) -> cp.ndarray:
    # Initiate constant vector
    return cp.array([const] * i)

def OP58(const: float, i: int, j: int) -> cp.ndarray:
    # Initiate constant matrix
    return ([[const]*j]*i)

def OP59(a:float = -1, b:float = 1) -> float:
    # generate a random scalar from uniform(a, b)
    return (cp.random.uniform(low=min(a, b), high=max(a, b)))

def OP60(a: float, b: float, i: int) -> cp.ndarray:
    # generate a random vector from uniform(a, b)
    return (cp.random.uniform(low=min(a, b), high=max(a, b), size=(i,)))

def OP61(a: float, b: float, i: int, j: int) -> cp.ndarray:
    # generate a random matrix from uniform(a, b)
    return (cp.random.uniform(low=min(a, b), high=max(a, b), size=(i, j)))

def OP62(mean: float = 0, std: float = 1) -> float:
    # generate a random scalar from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (cp.random.normal(loc = mean, scale = std))

def OP63(mean: float, std: float, i: int) -> cp.ndarray:
    # generate a random vector from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (cp.random.normal(loc = mean, scale = std, size = (i,)))

def OP64(mean: float, std: float, i: int, j: int) -> cp.ndarray:
    # generate a random matrix from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (cp.random.normal(loc = mean, scale = std, size = (i, j)))

def OP65(df: pd.DataFrame) -> cp.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' column to represent the value of scalar in all companies in order
        
    Returns
    -------
    cp.ndarray
        The ranking array of df.Scalar. If tie, return the average of ranks.

    '''
    return rankdata(df['Scalar'])

def OP66(df: pd.DataFrame) -> cp.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' and 'Industry' columns to represent the value of scalar and their Industry in all companies in order
        
    Returns
    -------
    cp.ndarray
        The relational ranking array of df.Scalar grouped by df.Industry. If tie, return the average of ranks.

    '''
    # define function to aggregate after grouping by Industry
    def rankIndustry(df: pd.DataFrame):
        df['rank'] = rankdata(df['Scalar'])
        return df
    return cp.array(df.groupby('Industry').apply(rankIndustry)['rank'], dtype = cp.float32)
    
def OP67(df: pd.DataFrame) -> cp.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' and 'Industry' columns to represent the value of scalar and their Industry in all companies in order
        
    Returns
    -------
    cp.ndarray
        The deviation array from the means of each df.Industry

    '''
    # define function to aggregate after grouping by Industry
    def deviate(df: pd.DataFrame):
        df['Deviate'] = df['Scalar'] - sum(df['Scalar'])/len(df['Scalar'])
        return df
    return cp.array(df.groupby('Industry').apply(deviate)['Deviate'], dtype = cp.float32)
