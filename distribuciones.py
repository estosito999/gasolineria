"""Prueba exploratoria de Poisson para tablas de frecuencias de valores exactos.

Este módulo solo analiza el ajuste; no elige la distribución de la simulación.
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2, poisson


def agrupar_clases(observados, esperados, etiquetas, minimo=5):
    """Fusiona clases contiguas según E, incluido el sobrante de la cola.

    Si la expectativa total no alcanza el mínimo, se devuelve una única clase;
    el llamador debe comprobar las condiciones de la prueba antes de aplicarla.
    """
    if not (len(observados) == len(esperados) == len(etiquetas)):
        raise ValueError("Observados, esperados y etiquetas deben tener igual longitud.")
    if not np.isfinite(minimo) or minimo <= 0:
        raise ValueError("El mínimo esperado debe ser positivo y finito.")
    for valores in (observados, esperados):
        valores = np.asarray(valores, dtype=float)
        if valores.ndim != 1 or not np.all(np.isfinite(valores)) or np.any(valores < 0):
            raise ValueError("Las frecuencias deben ser finitas y no negativas.")

    grupos = []
    obs_acumulada = 0
    esp_acumulada = 0.0
    etiquetas_acumuladas = []

    for observado, esperado, etiqueta in zip(observados, esperados, etiquetas):
        obs_acumulada += observado
        esp_acumulada += esperado
        etiquetas_acumuladas.append(str(etiqueta))

        if esp_acumulada >= minimo:
            grupos.append([
                obs_acumulada, esp_acumulada, " + ".join(etiquetas_acumuladas)
            ])
            obs_acumulada = 0
            esp_acumulada = 0.0
            etiquetas_acumuladas = []

    if etiquetas_acumuladas:
        if grupos:
            grupos[-1][0] += obs_acumulada
            grupos[-1][1] += esp_acumulada
            grupos[-1][2] += " + " + " + ".join(etiquetas_acumuladas)
        else:
            grupos.append([
                obs_acumulada, esp_acumulada, " + ".join(etiquetas_acumuladas)
            ])

    return grupos


def ajuste_poisson(df, alpha=0.05):
    """Contrasta H0: los valores registrados son compatibles con Poisson(λ).

    ``tiempo`` contiene valores enteros exactos, no intervalos ni marcas de
    clase; ``frecuencia`` contiene sus conteos. Se estima λ mediante la media
    ponderada de la misma muestra. La prueba asintótica supone observaciones
    independientes, agrupa hasta E >= 5 y usa gl = k - 1 - 1 (λ estimado).
    Un resultado inválido conserva la tabla y explica por qué no se emite p.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha debe estar entre 0 y 1, sin incluir los extremos.")
    if not {"tiempo", "frecuencia"}.issubset(df.columns) or df.empty:
        raise ValueError("Se requiere una tabla no vacía con tiempo y frecuencia.")

    try:
        valores = df[["tiempo", "frecuencia"]].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Los tiempos y frecuencias deben ser numéricos.") from exc
    if (not np.all(np.isfinite(valores)) or np.any(valores < 0)
            or np.any(valores != np.floor(valores))):
        raise ValueError("Los tiempos y frecuencias deben ser enteros no negativos y finitos.")

    # Acumular categorías repetidas sin expandir los datos ni alterar df.
    mapa_observado = {}
    for tiempo, frecuencia in valores:
        x, f = int(tiempo), int(frecuencia)
        if f:
            mapa_observado[x] = mapa_observado.get(x, 0) + f
    n = sum(mapa_observado.values())
    if n == 0:
        raise ValueError("La frecuencia total debe ser mayor que cero.")

    media = sum(x * f for x, f in mapa_observado.items()) / n
    max_x = max(mapa_observado)
    xs = np.arange(max_x + 1)

    # Incluir ceros observados y toda la cola: sum(E) = sum(O) = N.
    observados = [mapa_observado.get(int(x), 0) for x in xs] + [0]
    probabilidades = np.append(poisson.pmf(xs, media), poisson.sf(max_x, media))
    esperados = n * probabilidades
    etiquetas = [str(x) for x in xs] + [f">={max_x + 1}"]
    minimo_esperado = 5
    grupos = agrupar_clases(observados, esperados, etiquetas, minimo_esperado)
    tabla = pd.DataFrame(grupos, columns=["observada", "esperada", "clase"])
    tabla["contribucion"] = (
        (tabla["observada"] - tabla["esperada"]) ** 2
        / tabla["esperada"].where(tabla["esperada"] > 0)
    )

    parametros_estimados = 1
    grados_libertad = len(tabla) - 1 - parametros_estimados
    resultado = {
        "valido": False,
        "motivo": "",
        "n": n,
        "alpha": float(alpha),
        "lambda_estimado": float(media),
        "parametros_estimados": parametros_estimados,
        "minimo_esperado": minimo_esperado,
        "clases_originales": len(observados),
        "gl": grados_libertad,
        "chi2": None,
        "pvalor": None,
        "rechazar": None,
        "tabla": tabla,
    }

    if media == 0:
        resultado["motivo"] = (
            "λ estimado es cero: la Poisson es degenerada y no permite aplicar "
            "esta aproximación chi-cuadrado."
        )
    elif (tabla["esperada"] < minimo_esperado).any():
        resultado["motivo"] = (
            "No se alcanza Eᵢ ≥ 5, incluso después de agrupar las categorías."
        )
    elif grados_libertad <= 0:
        resultado["motivo"] = (
            "Quedan menos de tres clases después de agrupar. Al descontar "
            "el parámetro λ estimado, los grados de libertad no son positivos."
        )
    elif not np.isclose(tabla["esperada"].sum(), n):
        resultado["motivo"] = "Las frecuencias esperadas no suman el total observado."
    elif not np.all(np.isfinite(tabla["contribucion"])):
        resultado["motivo"] = "No se puede calcular una contribución finita para cada clase."
    else:
        estadistico = float(tabla["contribucion"].sum())
        pvalor = float(chi2.sf(estadistico, grados_libertad))
        resultado.update(
            valido=True,
            chi2=estadistico,
            pvalor=pvalor,
            rechazar=bool(pvalor < alpha),
        )

    return resultado
