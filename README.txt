---------------------------------------------------
RUN:
To run the AlphaEvolve, just run the file 'Run.py'.

---------------------------------------------------
To run the list of all 500 symbols: 
	+ go to 'GetData.py'
	+ go to line 64
	+ comment the last part
	+ save it and run 'Run.py'

---------------------------------------------------
To understand the code, I recommend read comments in 
the files in the following order:
1. Operands.py, OPs.py
2. Graph.py
3. Alpha.py
4. GetData.py, Features.py, Backtest.py
5. AlphaEvolve.py

----------------------------------------------------
Notes:
1 operation is written in form of:
	[Output: str, OP: int, Inputs: list]
Example:
	['v25', 60, [-1, 1, 13]]
	['s19', 50, ['v38']]
	['v49', 23, ['v6', 'v37']]
(OP is defined below)
----------------------------------------------------
Definition of Operation number:
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
