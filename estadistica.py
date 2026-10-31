import math

import numpy as np

from scipy.stats import t


# ============================================================
# INTERVALO DE CONFIANZA
# ============================================================

def intervalo_confianza(valores, confianza=0.95):

    valores = np.asarray(valores, dtype=float)

    n = len(valores)


    if n == 0:

        return {
            "n": 0,
            "media": np.nan,
            "desviacion": np.nan,
            "t_critico": np.nan,
            "semiancho": np.nan,
            "li": np.nan,
            "ls": np.nan,
            "error_relativo": np.nan
        }


    media = float(
        np.mean(valores)
    )


    # --------------------------------------------------------
    # NECESITAMOS AL MENOS 2 RÉPLICAS
    # --------------------------------------------------------

    if n < 2:

        return {
            "n": n,
            "media": media,
            "desviacion": np.nan,
            "t_critico": np.nan,
            "semiancho": np.nan,
            "li": np.nan,
            "ls": np.nan,
            "error_relativo": np.nan
        }


    # --------------------------------------------------------
    # DESVIACIÓN DE LAS MEDIAS DE LAS RÉPLICAS
    # --------------------------------------------------------

    desviacion = float(
        np.std(
            valores,
            ddof=1
        )
    )


    # --------------------------------------------------------
    # NIVEL DE SIGNIFICANCIA
    # --------------------------------------------------------

    alpha = 1 - confianza


    # --------------------------------------------------------
    # t DE STUDENT
    # --------------------------------------------------------

    t_critico = float(
        t.ppf(
            1 - alpha / 2,
            df=n - 1
        )
    )


    # --------------------------------------------------------
    # SEMIANCHO DEL INTERVALO
    # --------------------------------------------------------

    semiancho = (
        t_critico
        *
        desviacion
        /
        math.sqrt(n)
    )


    # --------------------------------------------------------
    # LÍMITES
    # --------------------------------------------------------

    li = media - semiancho

    ls = media + semiancho


    # --------------------------------------------------------
    # ERROR RELATIVO
    # --------------------------------------------------------

    if media != 0:

        error_relativo = (
            semiancho
            /
            abs(media)
        )

    else:

        error_relativo = np.inf


    return {

        "n": n,

        "media": media,

        "desviacion": desviacion,

        "t_critico": t_critico,

        "semiancho": semiancho,

        "li": li,

        "ls": ls,

        "error_relativo": error_relativo
    }


# ============================================================
# CALCULAR RÉPLICAS REQUERIDAS
# ============================================================

def replicas_requeridas(media, desviacion, precision=0.10, confianza=0.95, minimo=2, maximo=10000):

    if np.isnan(media) or np.isnan(desviacion):

        return minimo


    if media == 0:

        return minimo


    if desviacion == 0:

        return minimo


    # --------------------------------------------------------
    # ERROR MÁXIMO PERMITIDO
    # --------------------------------------------------------

    objetivo = (
        precision
        *
        abs(media)
    )


    alpha = 1 - confianza


    n = minimo


    # --------------------------------------------------------
    # PROCESO ITERATIVO
    # --------------------------------------------------------

    for _ in range(100):

        t_critico = float(
            t.ppf(
                1 - alpha / 2,
                df=n - 1
            )
        )


        nuevo_n = math.ceil(
            (
                t_critico
                *
                desviacion
                /
                objetivo
            ) ** 2
        )


        nuevo_n = max(
            minimo,
            nuevo_n
        )


        nuevo_n = min(
            nuevo_n,
            maximo
        )


        if nuevo_n == n:

            break


        n = nuevo_n


    return n


# ============================================================
# RESUMEN ESTADÍSTICO
# ============================================================

def resumen_estadistico(valores, confianza=0.95, precision=0.10):

    ic = intervalo_confianza(
        valores,
        confianza
    )


    replicas = replicas_requeridas(
        ic["media"],
        ic["desviacion"],
        precision,
        confianza
    )


    cumple_precision = False


    if not np.isnan(
        ic["error_relativo"]
    ):

        cumple_precision = (
            ic["error_relativo"]
            <= precision
        )


    return {

        **ic,

        "precision_objetivo":
            precision,

        "cumple_precision":
            cumple_precision,

        "replicas_requeridas":
            replicas
    }