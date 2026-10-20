import numpy as np

## 1. NORMAS
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

def normaliza(X,p):
    y = []
    for x in X:
        y.append(x/norma(x, p))
    return y

# 2. Normas matriciales inducidas
def calcularAX(A, x):
    n,m = A.shape
    res = np.zeros(n)
    for i in range(n):
        for j in range(m):
            res[i] += A[i][j]*x[j]
    return res

def normaMatMC(A,q,p,Np):
    n,m = A.shape
    res = -1
    vector = np.zeros(m)
    for i in range(Np):
        v = np.random.uniform(-1,1,m)
        vNormalizado = v/norma(v, p)
        normaQdeAx = norma(calcularAX(A,vNormalizado), q)
        if normaQdeAx > res:
            res = normaQdeAx
            vector = np.array(vNormalizado)

    return [res, vector]

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



# 3. Condicionamiento de Matrices
def condMC(A,p, numeroInutil):
    # condA = ||A||p ||A^-1||p
    c1 = normaMatMC(A, p, p, 10000)
    invA = np.linalg.inv(A) 
    c2 = normaMatMC(invA, p, p, 10000)

    return c1[0]*c2[0]

def variaPerc(b, perc):
    n = len(b)
    k = np.random.uniform(1-perc/100,1+perc/100)
    for i in range(n):
        b[i] *= k
    return b

def errorRelativoNormaP(x,xMonio, p):
    return norma(x-xMonio,p)/norma(x, p)

def buscarPeorVariacion(A, p, perc, NP):
    # A inversible -> A@x=b -> x = A^-1 @ b
    n,m = A.shape
    invA = np.linalg.inv(A)
    erroresX = []
    erroresB = []

    maximoErrorX = -1
    bMax = np.zeros(n)
    bMonioMax = np.zeros(n)

    for i in range(NP):
        b = np.random.rand(n)
        bMonio = variaPerc(b)
        x = invA @ b
        xMonio = invA @ bMonio
        errorX = errorRelativoNormaP(x, xMonio, p)

        if errorX > maximoErrorX:
            maximoErrorX = errorX
            bMax = b
            bMonioMax = bMonio

        erroresX.append(errorX)
        erroresB.append(errorRelativoNormaP(b, bMonio, p))
        
    condA = condMC(A,p)

    return erroresB, erroresX, condA, maximoErrorX, bMax, bMonioMax

def condExacta(A, p):
  """
  Que devuelve el numero de condicion de A a partir de la formula de la ecuacion (1) usando la norma p.
  """
  n,m = A.shape
  invA = np.linalg.inv(A)
  
  return normaExacta(A, p) * normaExacta(invA, p)