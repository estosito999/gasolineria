import pandas as pd

from generador import transformar_uniforme


# ============================================================
# CONSTRUIR EJEMPLO DE INTERARRIBOS
# ============================================================

def construir_ejemplo_interarribos(uniformes, tabla_llegadas):

    filas = []

    tiempo_acumulado = 0


    for i, u in enumerate(uniformes, start=1):

        u = float(u)


        # ----------------------------------------------------
        # TRANSFORMAR U -> TIEMPO DE INTERARRIBO
        # ----------------------------------------------------

        interarribo = transformar_uniforme(
            u,
            tabla_llegadas
        )


        # ----------------------------------------------------
        # OBTENER FILA DE LA DISTRIBUCIÓN
        # ----------------------------------------------------

        fila_distribucion = tabla_llegadas[
            tabla_llegadas["tiempo"] == interarribo
        ].iloc[0]


        probabilidad = float(
            fila_distribucion["probabilidad"]
        )


        limite_inferior = float(
            fila_distribucion["limite_inferior"]
        )


        limite_superior = float(
            fila_distribucion["limite_superior"]
        )


        # ----------------------------------------------------
        # MOMENTO DE LLEGADA
        # T_i = T_{i-1} + A_i
        # ----------------------------------------------------

        tiempo_acumulado += interarribo


        filas.append({

            "vehiculo": i,

            "u": u,

            "interarribo": interarribo,

            "probabilidad": probabilidad,

            "limite_inferior": limite_inferior,

            "limite_superior": limite_superior,

            "llegada": tiempo_acumulado
        })


    return pd.DataFrame(filas)