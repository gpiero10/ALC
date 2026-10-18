### Funciones L05-QR
import numpy as np

def filtrarNumerosMenoresATolerancia(R, tol):
    m, n = R.shape
    for i in range(m):
        for j in range(n):
            if abs(R[i][j]) <= tol: R[i][j] = 0
    return

def transpuesta(A):
    A = np.array(A)
    n, m = A.shape
    transA = np.zeros((m,n))

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
    cantOps = n # n divisiones
    R[0][0] = norma_a1

    for j in range(1,n):
        q_j = columnasAccesiblesDeA[j] # q_j = colJesima(A), falta normalizar
        for k in range(0, j):
            qk = transQ[k]
            R[k][j] = productoPunto(qk, q_j) # n mul + n sumas = 2n
            q_j -= R[k][j]*qk # 2 opps
            cantOps += 2*n + 2

        R[j][j] = norma(q_j, 2) # n sumas + 2*n multiplicaciones + 1 (raiz)
        transQ[j] = q_j/R[j][j] # n divs
        cantOps += 4*n + 1

    filtrarNumerosMenoresATolerancia(R, tol)

    if retorna_nops: return transpuesta(transQ), R, cantOps    
    
    return transpuesta(transQ), R    

def calcularSigno(aNumber):
    if aNumber >= 0: return 1
    else: return -1  

def multiplicacionDeMatrices(A,B, tol):
    # retorna A @ B 
    n,m = A.shape
    x,y = B.shape

    r = np.zeros((n,y), dtype=np.float64)

    for i in range(n):
        for j in range(y):
            for e in range(x):
                r[i][j] += A[i][e] * B[e][j]
            if abs(r[i][j]) <= tol:  r[i][j] = 0

    return r

def QR_con_HH(A,tol=1e-12,extras=False):
    m,n = A.shape 
    if m<n: return None, None
    R = A.copy()
    Q = np.eye(m)

    diccionario = {'R_matrices' : [], 'Q_matrices': []}

    for k in range(n):
        x = R[k:m, k]
        alpha = -calcularSigno(x[0])*norma(x,2)
        vectorCanonico = np.zeros(m-k)
        vectorCanonico[0] = 1

        u = x - alpha*vectorCanonico
        normaDeU = norma(u,2)
        if normaDeU > tol:
            u = u/normaDeU
            houseHolderK = np.eye(m-k) - 2*multiplicacionDeMatrices(transpuesta(np.array([u])), np.array([u]), tol)
            houseHolderKMonio = np.eye(m)
            houseHolderKMonio[k:, k:] = houseHolderK

            if extras:
                diccionario['R_matrices'].append(R)
                diccionario['Q_matrices'].append(Q)

            R = multiplicacionDeMatrices(houseHolderKMonio, R, tol)
            Q = multiplicacionDeMatrices(Q, transpuesta(houseHolderKMonio), tol)
    
    filtrarNumerosMenoresATolerancia(R, tol)

    if extras:
        return Q, R, diccionario    
    
    return Q, R

def calculaQR(A,metodo='RH',tol=1e-12):
    """
    A una matriz de n x n 
    tol la tolerancia con la que se filtran elementos nulos en R    
    metodo = ['RH','GS'] usa reflectores de Householder (RH) o Gram Schmidt (GS) para realizar la factorizacion
    retorna matrices Q y R calculadas con Gram Schmidt (y como tercer argumento opcional, el numero de operaciones)
    Si el metodo no esta entre las opciones, retorna None
    """
    if metodo == 'RH':
        return QR_con_HH(A,tol)
    
    if metodo == 'GS':
        return QR_con_GS(A,tol)

# Tests L05-QR:

import numpy as np

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
check_QR(Q2,R2,A2)

Q3,R3 = QR_con_GS(A3)
check_QR(Q3,R3,A3)

Q4,R4 = QR_con_GS(A4)
check_QR(Q4,R4,A4)

# --- TESTS PARA QR_by_HH ---
Q2h,R2h = QR_con_GS(A2)
check_QR(Q2h,R2h,A2)

Q3h,R3h = QR_con_HH(A3)
check_QR(Q3h,R3h,A3)

Q4h,R4h = QR_con_HH(A4)
print(Q4h)
print(R4h)
print(multiplicacionDeMatrices(Q4h, R4h, 1e-10))
check_QR(Q4h,R4h,A4)

# --- TESTS PARA calculaQR ---
Q2c,R2c = calculaQR(A2,metodo='RH')
check_QR(Q2c,R2c,A2)

Q3c,R3c = calculaQR(A3,metodo='GS')
check_QR(Q3c,R3c,A3)

Q4c,R4c = calculaQR(A4,metodo='RH')
check_QR(Q4c,R4c,A4)