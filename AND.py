"""
Aprendizaje de perceptrón (neurona simple) para aproximar operación lógica AND

"""

import math

data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1)
]

# Inicialización
w1, w2 = 0.0, 0.0
b = 0.0        # Regular la activación final
alpha = 0.1

# Función sigmoide:
# (y > 0.5 -> 1)
# (y <= 0.5 -> 0)
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# Entrenamiento
for epoch in range(100):
    total_error = 0

    for x1, x2, y_real in data:
        # Forward
        z = w1 * x1 + w2 * x2 + b
        y_pred = sigmoid(z)

        error = (y_pred - y_real) ** 2
        total_error += error

        # Gradientes
        dL_dypred = 2 * (y_pred - y_real)
        dypred_dz = y_pred * (1 - y_pred)   # derivada sigmoide
        dz_dw1 = x1
        dz_dw2 = x2
        dz_db = 1

        # Gradiente total
        dL_dw1 = dL_dypred * dypred_dz * dz_dw1
        dL_dw2 = dL_dypred * dypred_dz * dz_dw2
        dL_db  = dL_dypred * dypred_dz * dz_db

        # Actualización
        w1 -= alpha * dL_dw1
        w2 -= alpha * dL_dw2
        b  -= alpha * dL_db

    if epoch % 99 == 0:
        print(f"Epoch {epoch}, Error total: {total_error:.4f}")

# Prueba final
print("\nResultados:")
for x1, x2, _ in data:
    z = w1 * x1 + w2 * x2 + b
    y_pred = sigmoid(z)
    if (y_pred > 0.5):
        y_pred = 1
    else:
        y_pred = 0
    print(f"{x1}, {x2} -> {y_pred:.4f}")
print(f"w1 = {w1}, w2 = {w2}, b = {b}")