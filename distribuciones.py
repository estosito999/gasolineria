import numpy as np
import pandas as pd

from scipy.stats import poisson
from scipy.stats import chi2


# ============================================================
# AGRUPAR CLASES CON FRECUENCIAS ESPERADAS PEQUEÑAS
# ============================================================

def agrupar_clases(observados, esperados, etiquetas, minimo=5):

    grupos = []

    obs_acumulada = 0

    esp_acumulada = 0

    etiquetas_acumuladas = []


    for observado, esperado, etiqueta in zip(observados, esperados, etiquetas):

        obs_acumulada += observado

        esp_acumulada += esperado

        etiquetas_acumuladas.append(str(etiqueta))


        if esp_acumulada >= minimo:

            grupos.append([
                obs_acumulada,
                esp_acumulada,
                " + ".join(etiquetas_acumuladas)
            ])


            obs_acumulada = 0

            esp_acumulada = 0

            etiquetas_acumuladas = []


    # --------------------------------------------------------
    # SOBRANTE
    # --------------------------------------------------------

    if etiquetas_acumuladas:

        if grupos:

            grupos[-1][0] += obs_acumulada

            grupos[-1][1] += esp_acumulada

            grupos[-1][2] += " + " + " + ".join(etiquetas_acumuladas)

        else:

            grupos.append([
                obs_acumulada,
                esp_acumulada,
                " + ".join(etiquetas_acumuladas)
            ])


    return grupos


# ============================================================
# AJUSTE POISSON
# ============================================================

def ajuste_poisson(df, alpha=0.05):

    n = int(
        df["frecuencia"].sum()
    )


    media = float(
        np.average(
            df["tiempo"],
            weights=df["frecuencia"]
        )
    )


    max_x = int(
        df["tiempo"].max()
    )


    mapa_observado = dict(
        zip(
            df["tiempo"],
            df["frecuencia"]
        )
    )


    xs = list(
        range(0, max_x + 1)
    )


    observados = [
        mapa_observado.get(x, 0)
        for x in xs
    ]


    # --------------------------------------------------------
    # COLA FINAL
    # --------------------------------------------------------

    observados.append(0)


    probabilidades = [
        poisson.pmf(x, media)
        for x in xs
    ]


    probabilidades.append(
        1 - poisson.cdf(max_x, media)
    )


    esperados = [
        n * p
        for p in probabilidades
    ]


    etiquetas = [
        str(x)
        for x in xs
    ]


    etiquetas.append(
        f">={max_x + 1}"
    )


    grupos = agrupar_clases(
        observados,
        esperados,
        etiquetas
    )


    tabla = pd.DataFrame(
        grupos,
        columns=[
            "observada",
            "esperada",
            "clase"
        ]
    )


    if len(tabla) <= 2:

        return {
            "valido": False,
            "tabla": tabla
        }


    chi2_calculado = (
        (
            tabla["observada"]
            -
            tabla["esperada"]
        ) ** 2
        /
        tabla["esperada"]
    ).sum()


    # --------------------------------------------------------
    # λ fue estimado
    # --------------------------------------------------------

    parametros_estimados = 1


    grados_libertad = (
        len(tabla)
        -
        1
        -
        parametros_estimados
    )


    if grados_libertad <= 0:

        return {
            "valido": False,
            "tabla": tabla
        }


    pvalor = chi2.sf(
        chi2_calculado,
        grados_libertad
    )


    return {

        "valido": True,

        "lambda_estimado": media,

        "chi2": float(chi2_calculado),

        "gl": grados_libertad,

        "pvalor": float(pvalor),

        "rechazar": pvalor < alpha,

        "tabla": tabla
    }