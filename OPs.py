import numpy as np
import pandas as pd
from scipy.stats import rankdata
from Operands import Scalar, Vector, Matrix
np.seterr(all="ignore")
pd.options.mode.chained_assignment = None
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


def OP1(s1: Scalar, s2: Scalar) -> np.float64:
    # s1 + s2
    return (s1.value + s2.value)


def OP2(s1: Scalar, s2: Scalar) -> np.float64:
    # s1 - s2
    return (s1.value - s2.value)


def OP3(s1: Scalar, s2: Scalar) -> np.float64:
    # s1 * s2
    return (s1.value*s2.value)


def OP4(s1: Scalar, s2: Scalar) -> np.float64:
    # s1 / s2
    if s2.value == 0:
        raise(inputError("Divisor cannot be 0"))
    return (s1.value/s2.value)


def OP5(s: Scalar) -> np.float64:
    # |s|
    return (abs(s.value))


def OP6(s: Scalar) -> np.float64:
    # 1/s
    if s.value == 0:
        raise(inputError("Cannot inverse 0"))
    return (1/s.value)


def OP7(s: Scalar) -> np.float64:
    # sin(s)
    return (np.sin(s.value))


def OP8(s: Scalar) -> np.float64:
    # cos(s)
    return (np.cos(s.value))


def OP9(s: Scalar) -> np.float64:
    # tan(s)
    return (np.tan(s.value))


def OP10(s: Scalar) -> np.float64:
    # arcsin(s)
    if abs(s.value) > 1:
        s.value = np.sign(s.value)
    return (np.arcsin(s.value))


def OP11(s: Scalar) -> np.float64:
    # arccos(s)
    if abs(s.value) > 1:
        s.value = np.sign(s.value)
    return (np.arccos(s.value))


def OP12(s: Scalar) -> np.float64:
    # arctan(s)
    return (np.arctan(s.value))


def OP13(s: Scalar) -> np.float64:
    # e^s
    return (np.exp(s.value))


def OP14(s: Scalar) -> np.float64:
    # ln(s)
    if s.value <= 0:
        raise(inputError("Input must be positive"))
    return (np.log(s.value))


def OP15(s: Scalar) -> np.float64:
    # heaviside(s) for scalar
    return (max(np.sign(s.value), 0))


def OP16(v: Vector) -> np.ndarray:
    # heaviside(v) for vector
    return (np.heaviside(v.value, 0))


def OP17(m: Matrix) -> np.ndarray:
    # heaviside(m) for matrix
    return (np.heaviside(m.value, 0))


def OP18(s: Scalar, v: Vector) -> np.ndarray:
    # s*v
    return (s.value*v.value)


def OP19(s: Scalar, i: int) -> np.ndarray:
    # v = bcast(s): scalar to vector
    # i: length of vector output
    return np.array([s.value]*i)


def OP20(v: Vector) -> np.ndarray:
    # 1/v: inverse of vector
    if (v.value == 0).any():
        raise(inputError("Element cannot be 0"))
    return (1/v.value)


def OP21(v: Vector) -> np.float64:
    # ||v||
    return (np.linalg.norm(v.value))


def OP22(v: Vector) -> np.ndarray:
    # |v|
    return (abs(v.value))


def OP23(v1: Vector, v2: Vector) -> np.ndarray:
    # v1+v2
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value + v2.value)


def OP24(v1: Vector, v2: Vector) -> np.ndarray:
    # v1-v2
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value - v2.value)


def OP25(v1: Vector, v2: Vector) -> np.ndarray:
    # v1*v2
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (v1.value * v2.value)


def OP26(v1: Vector, v2: Vector) -> np.ndarray:
    # v1/v2
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors input must have the same size")
    if (v2.value == 0).any():
        raise inputError("Divisor input cannot be 0")
    return (v1.value / v2.value)


def OP27(v1: Vector, v2: Vector) -> np.float64:
    # dot product of 2 vectors
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors input must have the same size")
    return (np.dot(v1.value, v2.value))


def OP28(v1: Vector, v2: Vector) -> np.ndarray:
    # outer product
    return (np.outer(v1.value, v2.value))


def OP29(s: Scalar, m: Matrix) -> np.ndarray:
    # s*m
    return (s.value*m.value)


def OP30(m: Matrix) -> np.ndarray:
    # 1/m
    if (m.value == 0).any():
        raise(inputError("Element cannot be 0"))
    return (1/m.value)


def OP31(m: Matrix, v: Vector) -> np.ndarray:
    # dot(m, v)
    if m.value.shape[1] != len(v.value):
        raise inputError("Vector input must have the same size as row size of matrix")
    return np.dot(m.value, v.value)


def OP32(v: Vector, i: int) -> np.ndarray:
    # [[1, 2, 3, 4],
    # [1, 2, 3, 4],
    # [1, 2, 3, 4]]
    return np.array([v.value]*i)


def OP33(v: Vector, j: int) -> np.ndarray:
    # [[1, 1, 1],
    # [2, 2, 2],
    # [3, 3, 3],
    # [4, 4, 4]]
    return (np.transpose(np.array([v.value]*j)))


def OP34(m: Matrix) -> np.float64:
    # ||m||
    return (np.linalg.norm(m.value))


def OP35(m: Matrix) -> np.ndarray:
    # v[i] = norm(m[i,])
    return (np.linalg.norm(m.value, axis=1))


def OP36(m: Matrix) -> np.ndarray:
    # v[j] = norm(m[,j])
    return (np.linalg.norm(m.value, axis=0))


def OP37(m: Matrix) -> np.ndarray:
    # transpose(m)
    return (np.transpose(m.value))


def OP38(m: Matrix) -> np.float64:
    # |m|
    return (abs(m.value))


def OP39(m1: Matrix, m2: Matrix) -> np.ndarray:
    # m1+m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value + m2.value)


def OP40(m1: Matrix, m2: Matrix) -> np.ndarray:
    # m1-m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value - m2.value)


def OP41(m1: Matrix, m2: Matrix) -> np.ndarray:
    # m1*m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    return (m1.value * m2.value)


def OP42(m1: Matrix, m2: Matrix) -> np.ndarray:
    # m1/m2
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same size")
    if (m2.value == 0).any():
        raise(inputError("Element cannot be 0"))
    return (m1.value / m2.value)


def OP43(m1: Matrix, m2: Matrix) -> np.ndarray:
    # matmul(m1, m2)
    return (np.matmul(m1.value, m2.value))


def OP44(s1: Scalar, s2: Scalar) -> np.float64:
    # min(s1, s2)
    return (min(s1.value, s2.value))


def OP45(v1: Vector, v2: Vector) -> np.ndarray:
    # min(v1, v2)
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors must have the same size")
    return (np.minimum(v1.value, v2.value))


def OP46(m1: Matrix, m2: Matrix) -> np.ndarray:
    # min(m1, m2)
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same shape")
    return (np.minimum(m1.value, m2.value))


def OP47(s1: Scalar, s2: Scalar) -> np.float64:
    # max(s1, s2)
    return (max(s1.value, s2.value))


def OP48(v1: Vector, v2: Vector) -> np.ndarray:
    # max(v1, v2)
    if len(v1.value) != len(v2.value):
        raise inputError("Vectors must have the same size")
    return (np.maximum(v1.value, v2.value))


def OP49(m1: Matrix, m2: Matrix) -> np.ndarray:
    # max(m1, m2)
    if m1.value.shape != m2.value.shape:
        raise inputError("Matrices must have the same shape")
    return (np.maximum(m1.value, m2.value))


def OP50(v: Vector) -> np.float64:
    # mean(v)
    return (np.mean(v.value))


def OP51(m: Matrix) -> np.float64:
    # mean(m)
    return (np.mean(m.value))


def OP52(m: Matrix) -> np.ndarray:
    # vector mean of each row
    return (np.mean(m.value, axis=1))


def OP53(m: Matrix) -> np.ndarray:
    # vector std of each row
    return (np.std(m.value, axis=1))


def OP54(v: Vector) -> np.float64:
    # std of vector
    return (np.std(v.value))


def OP55(m: Matrix) -> np.float64:
    # std of matrix
    return (np.std(m.value))


def OP56(const: float) -> float:
    # Initiate constant scalar
    return const


def OP57(const: float, i: int) -> np.ndarray:
    # Initiate constant vector
    return np.array([const]*i)


def OP58(const: float, i: int, j: int) -> np.ndarray:
    # Initiate constant matrix
    return ([[const]*j]*i)


def OP59(a: float = -1, b: float = 1) -> float:
    # generate a random scalar from uniform(a, b)
    return (np.random.uniform(low=min(a, b), high=max(a, b)))


def OP60(a: float, b: float, i: int) -> np.ndarray:
    # generate a random vector from uniform(a, b)
    return (np.random.uniform(low=min(a, b), high=max(a, b), size=(i,)))


def OP61(a: float, b: float, i: int, j: int) -> np.ndarray:
    # generate a random matrix from uniform(a, b)
    return (np.random.uniform(low=min(a, b), high=max(a, b), size=(i, j)))


def OP62(mean: float = 0, std: float = 1) -> float:
    # generate a random scalar from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (np.random.normal(loc=mean, scale=std))


def OP63(mean: float, std: float, i: int) -> np.ndarray:
    # generate a random vector from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (np.random.normal(loc=mean, scale=std, size=(i,)))


def OP64(mean: float, std: float, i: int, j: int) -> np.ndarray:
    # generate a random matrix from normal distribution
    if std < 0:
        raise inputError("Standard deviation cannot be negative")
    return (np.random.normal(loc=mean, scale=std, size=(i, j)))


def OP65(df: pd.DataFrame) -> np.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' column to represent the value of scalar in all companies in order

    Returns
    -------
    np.ndarray
        The ranking array of df.Scalar. If tie, return the average of ranks.

    '''
    return rankdata(df['Scalar'])


def OP66(df: pd.DataFrame) -> np.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' and 'Industry' columns to represent the value of scalar and their Industry in all companies in order

    Returns
    -------
    np.ndarray
        The relational ranking array of df.Scalar grouped by df.Industry. If tie, return the average of ranks.

    '''
    # define function to aggregate after grouping by Industry
    def rankIndustry(df: pd.DataFrame):
        df['rank'] = rankdata(df['Scalar'])
        return df
    return np.array(df.groupby('Industry').apply(rankIndustry)['rank'], dtype=np.float64)


def OP67(df: pd.DataFrame) -> np.ndarray:
    '''

    Parameters
    ----------
    df : pd.DataFrame
        dataframe must have 'Scalar' and 'Industry' columns to represent the value of scalar and their Industry in all companies in order

    Returns
    -------
    np.ndarray
        The deviation array from the means of each df.Industry

    '''
    # define function to aggregate after grouping by Industry
    def deviate(df: pd.DataFrame):
        df['Deviate'] = df['Scalar'] - sum(df['Scalar'])/len(df['Scalar'])
        return df
    return np.array(df.groupby('Industry').apply(deviate)['Deviate'], dtype=np.float64)
