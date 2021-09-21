from copy import deepcopy

mat = [[5, 1, 0, 3, 1],
       [2, 5, -1, 1, -5],
       [3, -1, -7, 2, -5],
       [-1, 2, 3, 10, -16]]

b = [1, -5, -5, -16]

mat_nova = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
for linha in range(4):
    for coluna in range(5):
        mat_nova[linha][coluna] = mat[linha][coluna]/mat[linha][linha]

print(mat_nova)

xnovo = [0, 0, 0, 0]
for linha in range(4):
    xnovo[linha] = mat_nova[linha][-1]

erro = 1
xn = [0,0,0,0]
while erro >= 0.01:
    xn = deepcopy(xnovo)
    xnovo[0] = -1*mat_nova[0][1]*xn[1] -mat_nova[0][2]*xn[2] -mat_nova[0][3]*xn[3] + mat_nova[0][-1]
    xnovo[1] = -1*mat_nova[1][0]*xn[0] -mat_nova[1][2]*xn[2] -mat_nova[1][3]*xn[3] + mat_nova[1][-1]
    xnovo[2] = -1*mat_nova[2][0]*xn[0] -mat_nova[2][1]*xn[1] -mat_nova[2][3]*xn[3] + mat_nova[2][-1]
    xnovo[3] = -1*mat_nova[3][0]*xn[0] -mat_nova[3][1]*xn[1] -mat_nova[3][2]*xn[2] + mat_nova[3][-1]

    erro = 0
    for i in range(4):
       erro = max(erro, abs(xnovo[i]-xn[i]))

    print("X1={:0.3f} X2={:0.3f} X3={:0.3f} X4={:0.3f}".format(*xnovo))
    print("Erro={:0.1e}".format(erro))