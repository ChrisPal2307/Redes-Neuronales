"""
Aprendizaje de perceptrón multicapa para aproximar un intervalo
de una función no lineal (z = x * y)
"""

import random
import math

def tanh(x):
    return math.tanh(x)

def tanh_deriv(x):
    return 1 - math.tanh(x)**2

def sample():
    x1 = random.uniform(-1, 1)
    x2 = random.uniform(-1, 1)
    y = x1 * x2
    return x1, x2, y


# Pesos capa oculta (6 neuronas)
W = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(6)]
V = [random.uniform(-1, 1) for _ in range(6)]
B = [random.uniform(-1,1) for _ in range(6)]
b_out = random.uniform(-1,1)

lr = 0.005


for epoch in range(50000):
    x1, x2, y_real = sample()

    Z = []
    H = []
    for i in range(6):
        z = W[i][0]*x1 + W[i][1]*x2 + B[i]
        h = tanh(z)
        Z.append(z)
        H.append(h)

    y_pred = sum(V[i] * H[i] for i in range(6)) + b_out

    #Backprop
    dL_dy = 2 * (y_pred - y_real)
    dV = [dL_dy * H[i] for i in range(6)]
    dW = []
    dB = []
    for i in range(6):
        dh = dL_dy * V[i]
        dz = dh * tanh_deriv(Z[i])
        dW.append([dz * x1, dz * x2])
        dB.append(dz)

    db_out = dL_dy

    #Actualizacion
    for i in range(6):
        V[i] -= lr * dV[i]
        W[i][0] -= lr * dW[i][0]
        W[i][1] -= lr * dW[i][1]
        B[i] -= lr * dB[i]


print("Pruebas del modelo:")
tests = [(1, 1), (0.5, 0.5), (0.1, 0.2), (1.2, 0.8), (0.2, 0.9)]
for a, b in tests:
    Z = []
    H = []
    for i in range(6):
        z = W[i][0]*a + W[i][1]*b + B[i]
        h = tanh(z)
        Z.append(z)
        H.append(h)

    y_norm = sum(V[i] * H[i] for i in range(6))
    y_pred = y_norm + b_out

    print(f"{a} x {b} = {a*b:.4f} | Modelo = {y_pred:.4f}")



