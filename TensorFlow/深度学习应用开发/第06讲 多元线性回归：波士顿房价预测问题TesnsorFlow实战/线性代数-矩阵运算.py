#!/usr/bin/env python
# coding: utf-8

import numpy as np

# # 1 矩阵标量运算
#     如果矩阵乘，除，或加减一个标量，即：对矩阵的每一个元素进行数学运算
#     [1 2 3]       [2 4 5]
#             * 2 =
#     [4 5 6]       [8 10 12]
#
#     [a11 a12 a13]       [a11*n a12*n a13*n]
#                   * n =
#     [a21 a22 a23]       [a21*n a22*n a23*n]

# 标量只是一个单一的数字
scalar_value = 18
print(scalar_value)
scalar_np = np.array(scalar_value) # 转换为Numpy中的数组array
print(scalar_value,scalar_np.shape) # shape为()

#向量是一个有序的数字数组
vector_value = [1,2,3] # 这是一个列表
vector_np = np.array(vector_value) # 转换为Numpy中的数组array
print(vector_np,vector_np.shape) # shape 显示为一维数组，其实这既不能算行向量也不能算列向量

# 矩阵是一个有序的二维数组，它有两个王索引。第一个指向该行，第二个指向该列
matrix_list = [[1,2,3],[4,5,6]]
matrix_np = np.array(matrix_list)
print("matrix_list=",matrix_list,"\n","matrix_np=\n",matrix_np,"\n","matrix_np.shape=",matrix_np.shape)

#行向量的矩阵表示
vector_row = np.array([[1,2,3]])
print(vector_row,"shape=",vector_row.shape)

#列向量的矩阵表示
vector_column = np.array([[4],[5],[6]])

print(vector_column,"shape=",vector_column.shape)

#矩阵与标量运算
matrix_a = np.array([[1,2,3],[4,5,6]])
print(matrix_a,"shape=",matrix_a.shape)

#矩阵*标量
matrix_b = matrix_a*2
print(matrix_b,"shape=",matrix_b.shape)

#矩阵+标量
matrix_c = matrix_a+2
print(matrix_c,"shape=",matrix_c.shape)

# # 2 矩阵-矩阵加法和减法
#     矩阵-矩阵加法和减法要求是矩阵具有相同的尺寸，并且结果将是具有相同的尺寸的矩阵。
#     只需在第一个矩阵中添加或减去第二个矩阵的每个值及其对应的值
#     [a11 a12]   [b11 b12]   [a11+b11 a12+b12]
#               +           =
#     [a21 a22]   [b21 b22]   [a21+b21 a22+b22]

matrix_a = np.array([[1,2,3],
                     [4,5,6]]) #2行3列

matrix_b = np.array([[-1,-2,-3],
                     [-4,5,-6]]) #2行3列

print(matrix_a + matrix_b)

# # 3 矩阵-矩阵点乘(点乘)
#     矩阵-矩阵点乘要求是矩阵具有相同的尺寸，矩阵各个对应元素相乘
#     [a11 a12]   [b11 b12]   [a11*b11 a12*b12]
#               *           =
#     [a21 a22]   [b21 b22]   [a21*b21 a22*b22]
print(matrix_a * matrix_b)
print(np.multiply(matrix_a, matrix_b))


# # 4 矩阵-矩阵相乘(叉乘)
#     如果第一个矩阵列的数量与第二个矩阵行数要相等，才能将矩阵相乘结果矩阵具有与第一个矩阵相同的行数和第二个矩阵相同的列数
#     [a11 a12 a13]   [b11 b12]   [a11*b11+a12*b21+a13*b31 a11*b12+a12*b22+a13*b33]
#                   * [b21 b22] =
#     [a21 a22 a23]   [b31 b33]   [a21*b11+a22*b21+a23*b31 a21*b12+a22*b22+a23*b33]
#         A               B           A*B=C
#       (2,3)           (3,2)         (2,2)
#        m*n             n*k           m*k
matrix_a = np.array([[1,2,3],[4,5,6]])                  #2行3列
matrix_b = np.array([[1,2,3,4],[2,1,2,0],[3,4,1,2]])    #3行4列
print(np.matmul(matrix_a, matrix_b))                    #结果是2行4列


# # 5 矩阵-向量乘法
#     看作矩阵-矩阵叉乘的特列
#     [a11 a12 a13]   [b11]   [a11*b11+a12*b21+a13*b31]
#                   * [b21] =
#     [a21 a22 a23]   [b31]   [a21*b11+a22*b21+a23*b31]
#         A             B          A*B=C
#       (2,3)         (3,1)        (2,1)
#        m*n           n*1          m*1
matrix_a = np.array([[1,2,3],[4,5,6]])  #2行3列
matrix_b = np.array([[1],[2],[3]])      #3行1列
print(np.matmul(matrix_a, matrix_b))    #结果是2行1列

# # 6 向量-向量乘法(列向量-行向量)
#     看作矩阵-矩阵叉乘的特列
#     [b11]               [b11*a11 b11*a12]
#     [b21] * [a11 a12] = [b21*a11 b21*a12]
#     [b31]               [b31*a11 b31*a12]
#       B         A           B*A=C
#     (3,1)     (1,2)         (3,2)
#      n*1       1*m           n*m
matrix_a = np.array([[1],[2]])       #2行1列
matrix_b = np.array([[1,2]])         #1行2列
print(np.matmul(matrix_a, matrix_b)) #结果是2行2列


# # 7 向量-向量乘法(行向量-列向量)
#     看作矩阵-矩阵叉乘的特例中的特例
#                     [b11]
#     [a11 a12 a13] * [b21] = [a11*b11+a12*b21+a13*b31]
#                     [b31]
#           B           A           B*A=C
#         (1,3)       (3,1)         (1,1)
#          1*n         m*1           1*1
matrix_a = np.array([[1,2,3]])       #1行3列
matrix_b = np.array([[1],[2],[3]])   #3行1列
print(np.matmul(matrix_a, matrix_b)) #结果是1行1列

# # 8 矩阵转置
#     第一列变成转置矩阵的第一行，第二列变成了矩阵转置的第二行，一个m*n矩阵被转换成一个n*m矩阵
#     a的a[i][j]元素等于转置矩阵aT的a[j][i]元素
#        a        [1] aT
#     [1 2 3] ->  [2]
#                 [3]
#     转置矩阵像沿着45度轴线的矩阵镜像
#        b
#     [1 2 3] -> [1 4] bT
#                [2 5]
#     [4 5 6] -> [3 6]

#矩阵向量乘法
#将矩阵与矢量相乘可以被认为将矩阵的每一行与矢量的列相乘，矩阵的列数必须等于向量的行
#输出将是一个具体与矩阵相同行数的向量。
matrix_a = np.array([[1,2,3],[4,5,6]])  #2行3列
matrix_b = np.array([[1],[2],[3]])      #3行1列
print(np.matmul(matrix_a,matrix_b))     #结果是2行1列

matrix_a = np.array([[1,2,3]])       #1行3列
matrix_b = np.array([[2],[4],[-1]])  #3行1列
print(np.matmul(matrix_a,matrix_b))  #结果是1行1列

#矩阵转置
matrix_a = np.array([[1,2,3],[4,5,6]])
print(matrix_a,".shape=",matrix_a.shape,"\n",matrix_a.T,".Tshap=",matrix_a.T.shape)

#行列转置
vector_row = np.array([[1,2,3]])
print(vector_row,".shape=",vector_row.shape,"\n",vector_row.T,".Tshape=",vector_row.T.shape)

#reshape
vector_row = np.array([[1,2,3]])
vector_column = vector_row.reshape(3,1)
print(vector_row,".shape=",vector_row.shape,"\n",vector_column,".shape=",vector_column.shape)


