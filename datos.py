import numpy as np
import pandas as pd


# ============================================================
# TIEMPOS ENTRE LLEGADAS
# ============================================================

LLEGADAS = pd.DataFrame({
    "tiempo": [1, 2, 3, 4, 5],
    "frecuencia": [10, 15, 15, 7, 3]
})


# ============================================================
# TIEMPOS DE SERVICIO
# ============================================================

SERVICIO = pd.DataFrame({
    "tiempo": [2, 3, 4, 5, 6],
    "frecuencia": [5, 10, 20, 10, 5]
})


# ============================================================
# CONSTRUIR DISTRIBUCIÓN EMPÍRICA
# ============================================================

def preparar_distribucion(df):

    tabla = df.copy()

    total = tabla["frecuencia"].sum()

    tabla["probabilidad"] = tabla["frecuencia"] / total

    tabla["acumulada"] = tabla["probabilidad"].cumsum()

    tabla["limite_inferior"] = tabla["acumulada"].shift(1, fill_value=0.0)

    tabla["limite_superior"] = tabla["acumulada"]

    return tabla


# ============================================================
# MEDIA PONDERADA
# ============================================================

def media_ponderada(df):

    return float(np.average(df["tiempo"], weights=df["frecuencia"]))


# ============================================================
# RESUMEN GENERAL
# ============================================================

def resumen_datos():

    media_llegadas = media_ponderada(LLEGADAS)

    media_servicio = media_ponderada(SERVICIO)

    lambda_min = 1 / media_llegadas

    mu_min = 1 / media_servicio

    rho = lambda_min / mu_min

    return {
        "media_entre_llegadas": media_llegadas,
        "media_servicio": media_servicio,

        "lambda_min": lambda_min,
        "mu_min": mu_min,

        "lambda_hora": lambda_min * 60,
        "mu_hora": mu_min * 60,

        "rho": rho
    }


# ============================================================
# OBTENER DISTRIBUCIONES
# ============================================================

def obtener_distribuciones():

    tabla_llegadas = preparar_distribucion(LLEGADAS)

    tabla_servicio = preparar_distribucion(SERVICIO)

    return tabla_llegadas, tabla_servicio