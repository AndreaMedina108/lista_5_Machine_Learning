import numpy as np


def riesgo_cuadratico(y_true, y_pred):
    """
    Calcula el riesgo empírico cuadrático mediante función propia:
    R_S(h) = (1 / |S|) * sum_{i in S} (y_i - h(x_i))^2
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))