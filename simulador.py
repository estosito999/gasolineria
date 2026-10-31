import numpy as np
import pandas as pd

from datos import obtener_distribuciones
from generador import generar_tiempos


# ============================================================
# SIMULAR GASOLINERA
# ============================================================

def simular_gasolinera(n_clientes=100, semilla=2026):

    tabla_llegadas, tabla_servicio = obtener_distribuciones()

    rng = np.random.default_rng(semilla)


    # --------------------------------------------------------
    # GENERAR TIEMPOS
    # --------------------------------------------------------

    u_llegadas, entre_llegadas = generar_tiempos(n_clientes, tabla_llegadas, rng)

    u_servicios, servicios = generar_tiempos(n_clientes, tabla_servicio, rng)


    # --------------------------------------------------------
    # HORAS DE LLEGADA
    # --------------------------------------------------------

    llegadas = np.cumsum(entre_llegadas)


    filas = []

    salidas_previas = []


    # --------------------------------------------------------
    # SIMULACIÓN FIFO
    # --------------------------------------------------------

    for i in range(n_clientes):

        llegada = int(llegadas[i])

        servicio = int(servicios[i])


        if i == 0:

            salida_anterior = 0

        else:

            salida_anterior = salidas_previas[-1]


        # ----------------------------------------------------
        # INICIO DEL SERVICIO
        # ----------------------------------------------------

        inicio = max(llegada, salida_anterior)


        # ----------------------------------------------------
        # SALIDA
        # ----------------------------------------------------

        salida = inicio + servicio


        # ----------------------------------------------------
        # ESPERA
        # ----------------------------------------------------

        espera = inicio - llegada


        # ----------------------------------------------------
        # TIEMPO TOTAL EN EL SISTEMA
        # ----------------------------------------------------

        tiempo_sistema = salida - llegada


        # ----------------------------------------------------
        # CUÁNTOS VEHÍCULOS HABÍA EN EL SISTEMA
        # ----------------------------------------------------

        en_sistema = sum(1 for d in salidas_previas if d > llegada)

        cola_al_llegar = max(0, en_sistema - 1)


        filas.append({
            "vehiculo": i + 1,

            "u_llegada": float(u_llegadas[i]),

            "entre_llegadas": int(entre_llegadas[i]),

            "llegada": llegada,

            "u_servicio": float(u_servicios[i]),

            "servicio": servicio,

            "inicio_servicio": inicio,

            "salida": salida,

            "espera_cola": espera,

            "tiempo_sistema": tiempo_sistema,

            "cola_al_llegar": cola_al_llegar
        })


        salidas_previas.append(salida)


    return pd.DataFrame(filas)


# ============================================================
# LONGITUD DE COLA EN UN INSTANTE
# ============================================================

def longitud_cola_en_tiempo(df, t):

    condicion = (
        (df["llegada"] <= t)
        &
        (df["inicio_servicio"] > t)
    )

    return int(condicion.sum())


# ============================================================
# ESTADO COMPLETO EN EL TIEMPO t
# ============================================================

def estado_en_tiempo(df, t):

    esperando = df[
        (df["llegada"] <= t)
        &
        (df["inicio_servicio"] > t)
    ]["vehiculo"].tolist()


    atendiendo = df[
        (df["inicio_servicio"] <= t)
        &
        (df["salida"] > t)
    ]["vehiculo"].tolist()


    finalizados = df[
        df["salida"] <= t
    ]["vehiculo"].tolist()


    pendientes = df[
        df["llegada"] > t
    ]["vehiculo"].tolist()


    return {
        "esperando": esperando,
        "atendiendo": atendiendo,
        "finalizados": finalizados,
        "pendientes": pendientes
    }


# ============================================================
# RESUMEN DE LA SIMULACIÓN
# ============================================================

def resumen_simulacion(df):

    eventos = sorted(
        set(
            df["llegada"].tolist()
            +
            df["inicio_servicio"].tolist()
        )
    )


    cola_maxima = max(
        (
            longitud_cola_en_tiempo(df, t)
            for t in eventos
        ),
        default=0
    )


    horizonte = float(df["salida"].iloc[-1])


    tiempo_ocupado = float(df["servicio"].sum())


    utilizacion = tiempo_ocupado / horizonte


    return {

        "espera_media": float(df["espera_cola"].mean()),

        "espera_maxima": float(df["espera_cola"].max()),

        "tiempo_medio_sistema": float(df["tiempo_sistema"].mean()),

        "cola_maxima": cola_maxima,

        "utilizacion_observada": min(utilizacion, 1.0),

        "horizonte": horizonte
    }