import numpy as np

def transpuesta(A):
    A = np.array(A)
    n, m = A.shape
    transA = np.zeros((n,m))

    for x in range(n):
        for y in range(m):
            transA[y][x] = A[x][y]

    return transA

def norma(x,p):
    if p == 'inf':
        r = abs(x[0])
        for i in range(1,len(x)):
            r = max(r, abs(x[i]))
        return r
    else:            
        ac = 0
        for i in range(len(x)):
            ac += abs(x[i])**p
        return ac**(1/p)

def productoPunto(v1, v2):
    lenV1 = len(v1)
    lenV2 = len(v2)

    if lenV1 != lenV2: return None

    res = 0
    for i in range(lenV1):
        res += v1[i]*v2[i]

    return res 

def QR_con_GS(A,tol=1e-12,retorna_nops=False):
    n, m = A.shape
    if n != m: return None

    # Q sera llenada por fila, despues la transpongo -> easier
    transQ = np.zeros((n,n)) 

    R = np.zeros((n,n))

    columnasAccesiblesDeA = transpuesta(A)

    a1 = columnasAccesiblesDeA[0]
    norma_a1 = norma(a1, 2)

    transQ[0] = a1/norma_a1
    R[0][0] = norma_a1

    for j in range(1,n):
        q_j = columnasAccesiblesDeA[j] # q_j = colJesima(A), falta normalizar
        for k in range(0, j):
            qk = transQ[k]
            R[k][j] = productoPunto(qk, q_j)
            q_j -= R[k][j]*qk

        R[j][j] = norma(q_j, 2)
        transQ[j] = q_j/R[j][j]
    
    return transpuesta(transQ), R


# TESTS
# --- Matrices de prueba ---
A2 = np.array([[1., 2.],
               [3., 4.]])

A3 = np.array([[1., 0., 1.],
               [0., 1., 1.],
               [1., 1., 0.]])

A4 = np.array([[2., 0., 1., 3.],
               [0., 1., 4., 1.],
               [1., 0., 2., 0.],
               [3., 1., 0., 2.]])

# --- Funciones auxiliares para los tests ---
def check_QR(Q,R,A,tol=1e-10):
    # Comprueba ortogonalidad y reconstrucción
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=tol)
    assert np.allclose(Q @ R, A, atol=tol)

# --- TESTS PARA QR_by_GS2 ---
Q2,R2 = QR_con_GS(A2)
# print(Q2)
# print(R2)
# print(Q2@R2)
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)
