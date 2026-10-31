import pandas as pd


# ============================================================
# DATAFRAME -> CSV
# ============================================================

def dataframe_a_csv(df):

    return df.to_csv(
        index=False
    ).encode(
        "utf-8-sig"
    )


# ============================================================
# EXPORTAR SIMULACIÓN COMPLETA
# ============================================================

def exportar_simulacion(df):

    columnas = [

        "vehiculo",

        "u_llegada",

        "entre_llegadas",

        "llegada",

        "u_servicio",

        "servicio",

        "inicio_servicio",

        "salida",

        "espera_cola",

        "tiempo_sistema",

        "cola_al_llegar"
    ]


    columnas_existentes = [
        columna
        for columna in columnas
        if columna in df.columns
    ]


    resultado = df[
        columnas_existentes
    ].copy()


    return dataframe_a_csv(
        resultado
    )


# ============================================================
# EXPORTAR RESULTADOS DE RÉPLICAS
# ============================================================

def exportar_replicas(df):

    columnas = [

        "replica",

        "semilla",

        "n_clientes",

        "espera_media",

        "espera_maxima",

        "cola_maxima",

        "tiempo_medio_sistema",

        "utilizacion_observada",

        "horizonte"
    ]


    columnas_existentes = [
        columna
        for columna in columnas
        if columna in df.columns
    ]


    resultado = df[
        columnas_existentes
    ].copy()


    return dataframe_a_csv(
        resultado
    )


# ============================================================
# EXPORTAR PRIMEROS NÚMEROS ALEATORIOS
# ============================================================

def exportar_numeros_aleatorios(df, cantidad=5):

    columnas = [

        "vehiculo",

        "u_llegada",

        "entre_llegadas",

        "u_servicio",

        "servicio"
    ]


    resultado = df[
        columnas
    ].head(
        cantidad
    ).copy()


    return dataframe_a_csv(
        resultado
    )


# ============================================================
# EXPORTAR TRES VEHÍCULOS PARA VERIFICACIÓN MANUAL
# ============================================================

def exportar_verificacion_manual(df, inicio=0, cantidad=3):

    columnas = [

        "vehiculo",

        "entre_llegadas",

        "llegada",

        "servicio",

        "inicio_servicio",

        "salida",

        "espera_cola",

        "tiempo_sistema"
    ]


    resultado = df[
        columnas
    ].iloc[
        inicio:
        inicio + cantidad
    ].copy()


    return dataframe_a_csv(
        resultado
    )


# ============================================================
# GENERAR RESUMEN DE TEXTO
# ============================================================

def generar_resumen_txt(resumen_datos, resumen_simulacion, ic):

    texto = f"""
SIMULACIÓN DE GASOLINERA
========================

DATOS DEL SISTEMA
-----------------

Media entre llegadas:
{resumen_datos["media_entre_llegadas"]:.4f} min

Media de servicio:
{resumen_datos["media_servicio"]:.4f} min

Tasa de llegada:
{resumen_datos["lambda_hora"]:.4f} veh/h

Tasa de servicio:
{resumen_datos["mu_hora"]:.4f} veh/h

Rho:
{resumen_datos["rho"]:.4f}


RESULTADOS DE SIMULACIÓN
------------------------

Espera media:
{resumen_simulacion["espera_media"]:.4f} min

Espera máxima:
{resumen_simulacion["espera_maxima"]:.4f} min

Cola máxima:
{resumen_simulacion["cola_maxima"]}

Utilización observada:
{resumen_simulacion["utilizacion_observada"] * 100:.2f} %

Horizonte:
{resumen_simulacion["horizonte"]:.4f} min


INTERVALO DE CONFIANZA
----------------------

Media:
{ic["media"]:.4f}

Desviación:
{ic["desviacion"]:.4f}

Semiancho:
{ic["semiancho"]:.4f}

Límite inferior:
{ic["li"]:.4f}

Límite superior:
{ic["ls"]:.4f}

Error relativo:
{ic["error_relativo"] * 100:.2f} %
"""


    return texto.encode(
        "utf-8"
    )
