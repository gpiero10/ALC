import numpy as np

def norma(x, p):
    if p == 'inf':
        ans = 0
        for e in x:
            ans = max(ans, abs(e))
        return ans

    ans = 0
    for e in x:
        ans += abs(e) ** p
    ans = ans ** (1/p)
    return ans

def transponer(A):
    n, m = A.shape
    ans = np.zeros((m, n))
    for i in range(n):
        for j in range(m):
            ans[j][i] = A[i][j]
    return ans

def producto_matricial(A, B):
    n1, m1 = A.shape
    n2, m2 = B.shape
    assert(m1 == n2)
    ans = np.zeros((n1, m2))
    for i in range(n1):
        for j in range(m2):
            for k in range(m1):
                ans[i][j] += A[i][k] * B[k][j]
    return ans

def producto_matriz_por_vector(A, v):
    ans = np.array(v)
    v = np.array([v])
    v = transponer(v)
    prod = producto_matricial(A, v)
    for i in range(len(ans)):
        ans[i] = prod[i][0]
    return ans

def producto_interno(v, w):
    n = len(v)
    res = 0
    for i in range(n): res += v[i] * w[i]
    return res

def producto_vector_por_vector_transpuesto(v, w):
    n = len(v)
    A = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            A[i][j] = v[i] * w[j]

    return A

def f(A, v):
    ans = producto_matriz_por_vector(A, v)
    ans /= norma(ans, 2)
    return ans

def metpot2k(A, tol=1e-15, K=1000):
    n = len(A)
    v = np.random.uniform(-100, 100, n)
    w = f(A, f(A, v))
    k = 0
    e = producto_interno(w, v)

    while abs(e - 1) > tol and k < K:
        v = w
        w = f(A, f(A, v))
        e = producto_interno(w, v)
        k += 1

    autovector = w
    autovalor = producto_interno(w, producto_matriz_por_vector(A, w))
    return (autovector, autovalor, k)

def diagRH(A, tol=1e-15, K=1000):
    n = len(A)
    autovector1, autovalor1, _ = metpot2k(A, tol, K)
    e1 = np.zeros(n)
    e1[0] = 1

    H = producto_vector_por_vector_transpuesto(e1 - autovector1, e1 - autovector1)
    H = 2 * H / producto_interno(e1 - autovector1, e1 - autovector1)
    H = np.eye(n) - H

    if n == 2:
        S = H
        D = producto_matricial(H, producto_matricial(A, transponer(H)))
        return (S, D)

    B = producto_matricial(H, producto_matricial(A, transponer(H)))
    C = np.zeros((n-1,n-1))
    for i in range(n-1):
        for j in range(n-1):
            C[i][j] = B[i+1][j+1]

    S_aux, D_aux = diagRH(C, tol, K)
    D = np.zeros((n, n))
    S = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                D[i][j] = autovalor1
                continue
            if i == 0 or j == 0:
                D[i][j] = 0
                continue
            D[i][j] = D_aux[i-1][j-1]

    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                S[i][j] = 1
                continue
            if i == 0 or j == 0:
                S[i][j] = 0
                continue
            S[i][j] = S_aux[i-1][j-1]

    S = producto_matricial(H, S)
    return (S, D)

def svd_reducida(A, k="max",tol=1e-15):
	tol = 1e-14
	# print(A)
	prod = producto_matricial(transponer(A), A)
	S, D = diagRH(prod, tol)
    
	n, _ = D.shape
	r = 0
	for i in range(n):
		if D[i][i] > tol: r += 1
	
	# print(D[n-1][n-1])
	# print(n)
	
	# print("Aca AUTOVALORES y D")
	# print(r)
	# print(tol)
	# print(D)

	sigma = np.zeros((r, r))
	for i in range(r):
		sigma[i][i] = D[i][i] ** (1/2)

	# print(sigma)

	V = np.zeros((n,r))	
	for i in range(n):
		for j in range(r):
			V[i][j] = S[i][j]

	B = producto_matricial(A, V)

	sigma_inversa = sigma.copy()
	for i in range(r): sigma_inversa[i][i] = 1 / sigma_inversa[i][i]

	U = producto_matricial(B, sigma_inversa)
	#filas, columnas = U.shape
	#for j in range(columnas):
	#	x = U[:,j:j+1]
    #	val = norma(x, 2)
    #	U[:, j:j+1] /= val
	
	valores_singulares = np.zeros(r)
	for i in range(r): valores_singulares[i] = sigma[i][i]

	# print(valores_singulares)
	# print("voy a printear mi V transpuesto")
	# print(transponer(V))

	if k == "max": return (U, valores_singulares, V)
	U_truncada = U[:,0:k]
	valores_singulares_truncados = valores_singulares[0:k]
	# V_truncada = V[:,0:k]
	V_truncada = V[0:k,:]

	return (U_truncada, valores_singulares_truncados, V_truncada)
