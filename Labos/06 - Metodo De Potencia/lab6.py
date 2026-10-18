import numpy as np

def matricesIguales(A, B, atol=1e-8):
    res = True
    n1, m1 = A.shape
    n2, m2 = B.shape
    if n1 != n2 or n2 != m2: return False

    for x in range(n1):
        for y in range(m1):
            res &= (abs(A[x][y] - B[x][y]) <= atol)
    return res

def esSimetrica(A):
    return matricesIguales(A, transpuesta(A))

def calcularAX(A, x):
    # equiv A@x, donde x se pasa como vector fila
    n,m = A.shape
    res = np.zeros(n)
    for i in range(n):
        for j in range(m):
            res[i] += A[i][j]*x[j]
    return res

def productoPunto(v1, v2):
    lenV1 = len(v1)
    lenV2 = len(v2)

    if lenV1 != lenV2: return None

    res = 0
    for i in range(lenV1):
        res += v1[i]*v2[i]

    return res 

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

def multiplicacionDeMatrices(A,B, tol=1e-8):
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

def transformacionF_A(A, v):
    w_monio = calcularAX(A, v)
    norma2DeWMonio = norma(w_monio, 2)
    
    if norma2DeWMonio > 0: return w_monio/norma2DeWMonio
    else: return np.zeros(A.shape[0])

def metpot2k(A, tol= 1e-15, K = 1e3):
    n,m = A.shape
    v_0 = np.random.rand(n)
    v_monio = transformacionF_A(A, transformacionF_A(A, v_0)) # F_A^2(v)
    error = productoPunto(v_monio, v_0)
    k = 0

    while abs(error-1) > tol and k < K:
        v_0 = v_monio
        v_monio = transformacionF_A(A, transformacionF_A(A, v_0)) # F_A^2(v)
        error = productoPunto(v_monio, v_0)
        k += 1

    matrizAPorVMonio = calcularAX(A, v_monio)
    hansLanda = productoPunto(v_monio, matrizAPorVMonio)
    error = error - 1

    return v_0, hansLanda, k


def diagRH(A, tol = 1e-15, K=1000):
    n,m = A.shape

    if not esSimetrica(A): None, None

    e1 = np.zeros(n)
    e1[0] = 1
    vector, landa, _ = metpot2k(A, tol, K)
    vectorResta = e1 - vector
    normaVector = norma(vectorResta, 2)
    matriz_e1_menos_autovector = multiplicacionDeMatrices(transpuesta(np.array([vectorResta])), np.array([vectorResta]))
    matrizHouse = np.eye(n) - (2/normaVector**2)*matriz_e1_menos_autovector

    if n == 2:
        S = matrizHouse
        D = multiplicacionDeMatrices(S, multiplicacionDeMatrices(A, transpuesta(S)))
    else: 
        B = multiplicacionDeMatrices(matrizHouse, multiplicacionDeMatrices(A, transpuesta(matrizHouse)))
        A_monio = B[1:n,1:n]
        S_monio, D_monio = diagRH(A_monio, tol, K)
        D = np.eye(n)
        D[0][0] = landa
        D[1:n, 1:n] = D_monio
        factor = np.eye(n)
        factor[1:n, 1:n] = S_monio
        S = multiplicacionDeMatrices(matrizHouse, factor)

    return S, D

def calcularNormaExacta(A, p):
    n, m = A.shape
    res = None
    if p == 'inf':
        res = 0
        for i in range(n):
            filaIesimaSum = 0
            for j in range(m):
                filaIesimaSum += abs(A[i][j])
            res = max(res, filaIesimaSum)
    if p == 1:
        res = 0
        for i in range(m):
            columnaIesimaSum = 0
            for j in range(n):
                columnaIesimaSum += abs(A[j][i])
            res = max(res, columnaIesimaSum)
    return res

def normaExacta(A,p=[1,'inf']):
    # p = [1, inf]
    if isinstance(p, list):
        resLista = np.zeros(len(p))
        for i in range(len(p)):
            resLista[i] = calcularNormaExacta(A, p[i])
        return resLista
    else:
        return calcularNormaExacta(A, p)

#### TESTEOS
# Tests metpot2k

S = np.vstack([
    np.array([2,1,0])/np.sqrt(5),
    np.array([-1,2,5])/np.sqrt(30),
    np.array([1,-2,1])/np.sqrt(6)
              ]).T

# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    D = np.diag(np.random.random(3)+1)*100
    A = S@D@S.T
    v,l,_ = metpot2k(A,1e-15,1e5)
    if np.abs(l - np.max(D))< 1e-8:
        exitos += 1
assert exitos > 95


#Test con HH
exitos = 0
for i in range(100):
    v = np.random.rand(9)
    #v = np.abs(v)
    #v = (-1) * v
    ixv = np.argsort(-np.abs(v))
    D = np.diag(v[ixv])
    I = np.eye(9)
    H = I - 2*np.outer(v.T, v)/(np.linalg.norm(v)**2)   #matriz de HouseHolder

    A = H@D@H.T
    v,l,_ = metpot2k(A, 1e-15, 1e5)
    #max_eigen = abs(D[0][0])
    if abs(l - D[0,0]) < 1e-8:         
        exitos +=1
assert exitos > 95



# Tests diagRH
D = np.diag([1,0.5,0.25])
S = np.vstack([
    np.array([1,-1,1])/np.sqrt(3),
    np.array([1,1,0])/np.sqrt(2),
    np.array([1,-1,-2])/np.sqrt(6)
              ]).T

A = S@D@S.T
SRH,DRH = diagRH(A,tol=1e-15,K=1e5)
assert np.allclose(D,DRH)
assert np.allclose(np.abs(S.T@SRH),np.eye(A.shape[0]),atol=1e-7)



# Pedimos que pase el 95% de los casos
exitos = 0
for i in range(100):
    A = np.random.random((5,5))
    A = 0.5*(A+A.T)
    S,D = diagRH(A,tol=1e-15,K=1e5)
    ARH = S@D@S.T
    e = normaExacta(ARH-A,p='inf')
    if e < 1e-5: 
        exitos += 1
assert exitos >= 95
