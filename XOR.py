"""
Aprendizaje de perceptrón multicapa para aproximar operación lógica XOR

"""

import math
import random

# Datos XOR
data = [
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 1),
    (1, 1, 0)
]

# Inicialización aleatoria
w1, w2 = random.uniform(-1,1), random.uniform(-1,1)
w3, w4 = random.uniform(-1,1), random.uniform(-1,1)
w5, w6 = random.uniform(-1,1), random.uniform(-1,1)

b1, b2, b3 = random.uniform(-1,1), random.uniform(-1,1), random.uniform(-1,1)

alpha = 0.1

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Entrenamiento
for epoch in range(5000):
    total_error = 0

    for x1, x2, y_real in data:

        z1 = w1*x1 + w2*x2 + b1
        h1 = sigmoid(z1)

        z2 = w3*x1 + w4*x2 + b2
        h2 = sigmoid(z2)

        z3 = w5*h1 + w6*h2 + b3
        y_pred = sigmoid(z3)

        error = (y_pred - y_real)**2
        total_error += error

        # Backpropagation
        dL_dypred = 2 * (y_pred - y_real)
        dypred_dz3 = y_pred * (1 - y_pred)

        dL_dz3 = dL_dypred * dypred_dz3

        # Gradientes salida
        dL_dw5 = dL_dz3 * h1
        dL_dw6 = dL_dz3 * h2
        dL_db3 = dL_dz3

        # Propagar a capa oculta
        dL_dh1 = dL_dz3 * w5
        dL_dh2 = dL_dz3 * w6

        dh1_dz1 = h1 * (1 - h1)
        dh2_dz2 = h2 * (1 - h2)

        dL_dz1 = dL_dh1 * dh1_dz1
        dL_dz2 = dL_dh2 * dh2_dz2

        # Gradientes capa oculta
        dL_dw1 = dL_dz1 * x1
        dL_dw2 = dL_dz1 * x2
        dL_db1 = dL_dz1

        dL_dw3 = dL_dz2 * x1
        dL_dw4 = dL_dz2 * x2
        dL_db2 = dL_dz2

        # Actualización
        w1 -= alpha * dL_dw1
        w2 -= alpha * dL_dw2
        b1 -= alpha * dL_db1

        w3 -= alpha * dL_dw3
        w4 -= alpha * dL_dw4
        b2 -= alpha * dL_db2

        w5 -= alpha * dL_dw5
        w6 -= alpha * dL_dw6
        b3 -= alpha * dL_db3

    if epoch % 4999 == 0:
        print(f"Epoch {epoch}, Error: {total_error:.4f}")


print("\nResultados:")
for x1, x2, _ in data:
    z1 = w1*x1 + w2*x2 + b1
    h1 = sigmoid(z1)

    z2 = w3*x1 + w4*x2 + b2
    h2 = sigmoid(z2)

    z3 = w5*h1 + w6*h2 + b3
    y_pred = sigmoid(z3)
    if (y_pred > 0.5):
        y_pred = 1
    else:
        y_pred = 0
    print(f"{x1}, {x2} -> {y_pred:.4f}")
