#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eliminacion Gausianna
"""
import numpy as np

def transponer(A):
    transA = A.copy()
    m,n = A.shape
    for i in range(m):
        for j in range(n):
            transA[i][j] = A[j][i]

    return transA

def matrizConRangoCompleto(A):
    m,n = A.shape
    res = True
    for x in range(m):
        res &= (A[x][x] != 0)
    return res

def matricesIguales(A, B, atol=1e-8):
    res = True
    n1, m1 = A.shape
    n2, m2 = B.shape
    if n1 != n2 or n2 != m2: return False

    for x in range(n1):
        for y in range(m1):
            res &= (abs(A[x][y] - B[x][y]) <= atol)
    return res

def calculaLU(A):
    #agrego
    if A is None: return None, None, 0 
    
    cant_op = 0
    m=A.shape[0]
    n=A.shape[1]
    Ac = A.copy()
    
    if m!=n:
        print('Matriz no cuadrada')
        return None, None, 0
    
    ## desde aqui -- CODIGO A COMPLETAR
    L = np.eye(n)
    for x in range(m): #fila pivote
        for y in range(x+1, m):  # fila sobre la que se opera
            if Ac[x][x] == 0 or abs(Ac[x][x]) <= 1e-8: return None, None, 0 
            escalar = (-1)*(Ac[y][x]/Ac[x][x])
            cant_op += 1
            for z in range(x,n): # columna de la fila sobre la que se opera
                if x == z:
                    Ac[y][z] = 0
                    L[y][x] = (-1)*escalar
                else:
                    Ac[y][z] += Ac[x][z]*escalar
                    cant_op += 2
    U = Ac.copy()
    ## hasta aqui, calculando L, U y la cantidad de operaciones sobre 
    ## la matriz Ac
    return L, U, cant_op

def res_tri(L, b, inferior=True):
    n, m = L.shape
    solucion = np.zeros(m)
    
    if inferior:
        for x in range(n): #Fila, desde la primera hasta la ultima
            sumatoria = 0
            for y in range(x): #Columna
                sumatoria += solucion[y]*L[x][y]
                
            solucion[x] = (b[x]-sumatoria)/L[x][x]
        
    else:
        for x in range(n-1, -1, -1): #Fila, desde la ultima hasta la primera
            sumatoria = 0
            for y in range(n-1, x, -1): #Columna
                sumatoria += solucion[y]*L[x][y]
                
            solucion[x] = (b[x]-sumatoria)/L[x][x]
            
    return solucion

def inversa(A):
    n,m = A.shape

    L,U,cantOps = calculaLU(A)
    if matrizConRangoCompleto(U) == False: return None

    matrizIdentidad = np.eye(n)
    solucion = []
    
    for i in range(n):
        y = res_tri(L,matrizIdentidad[i],True)
        x = res_tri(U,y,False)
        solucion.append(x)

    return transponer(np.array(solucion))

def calculaLDV(A):
    L, U, n = calculaLU(A)
    V, D, m = calculaLU(transponer(U)) # V^t D = U^t

    return L,D,transponer(V)

def esSDP(A, atol=1e-8):
    esSimetrica = matricesIguales(transponer(A), A, atol)
    n,m = A.shape
    res = True
    L,D,V = calculaLDV(A)
    for x in range(n):
        res &= (D[x][x] > atol)

    return res and esSimetrica
