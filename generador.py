import numpy as np


# ============================================================
# TRANSFORMAR UN SOLO U
# ============================================================

def transformar_uniforme(u, tabla):

    acumulada = tabla["acumulada"].to_numpy()

    tiempos = tabla["tiempo"].to_numpy()

    indice = np.searchsorted(acumulada, u, side="left")

    indice = min(indice, len(tiempos) - 1)

    return int(tiempos[indice])


# ============================================================
# TRANSFORMAR MUCHOS U
# ============================================================

def transformar_vector(uniformes, tabla):

    acumulada = tabla["acumulada"].to_numpy()

    tiempos = tabla["tiempo"].to_numpy()

    indices = np.searchsorted(acumulada, uniformes, side="left")

    indices = np.clip(indices, 0, len(tiempos) - 1)

    return tiempos[indices].astype(int)


# ============================================================
# GENERAR NÚMEROS UNIFORMES
# ============================================================

def generar_uniformes(n, rng):

    return rng.random(n)


# ============================================================
# GENERAR TIEMPOS
# ============================================================

def generar_tiempos(n, tabla, rng):

    uniformes = generar_uniformes(n, rng)

    tiempos = transformar_vector(uniformes, tabla)

    return uniformes, tiempos