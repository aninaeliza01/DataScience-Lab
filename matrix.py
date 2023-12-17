import numpy as np

def matrix(name,rows,cols):
    matrix=[]
    print(f"The matrix {name}")
    for i in range(rows):
        row=[]
        for j in range(cols):
            element=int(input(f"Enter the row {i+1},column{j+1}:"))
            row.append(element)
        matrix.append(row)
    return np.array(matrix)

row=int(input(f"Enter the number of rows:"))
col=int(input(f"Enter the number of columns:"))
matrix_1=matrix("Matrix 1",row,col)
print(matrix_1)
matrix_2=matrix("Matrix 2",row,col)
print(matrix_2)

sum_matrix = np.add(matrix_1, matrix_2)
sub_matrix = np.subtract(matrix_1, matrix_2)
mul_matrix=np.multiply(matrix_1,matrix_2)
div_matrix=np.divide(matrix_1,matrix_2)
print("SUM\n", sum_matrix)
print("SUBTRACT\n", sub_matrix)
print("MULTIPICATION\n",mul_matrix)
print("DIVISION\n",div_matrix)
print("TRANSPOSE\n",np.transpose(matrix_1))