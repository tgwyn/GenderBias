import numpy as np
x = [2, 3, 4, 5, 6]
nums = np.array([2, 3, 4, 5, 6])
type(nums)

numpy.ndarray

array([2, 3, 4, 5, 6])

nums = np.array([[2,4,6], [8,10,12], [14,16,18]])


array([[ 2,  4,  6],
       [ 8, 10, 12],
       [14, 16, 18]])

nums = np.arange(2, 7)

zeros = np.zeros(5)

ones = np.ones(5)

a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])


row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)  # Prints "[5 6 7 8] (4,)"
print(row_r2, row_r2.shape)  # Prints "[[5 6 7 8]] (1, 4)"


col_r1 = a[:, 1]
col_r2 = a[:, 1:2]
print(col_r1, col_r1.shape)  # Prints "[ 2  6 10] (3,)"
print(col_r2, col_r2.shape) 
