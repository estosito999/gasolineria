import pandas as pd

from simulador import simular_gasolinera
from simulador import resumen_simulacion


# ============================================================
# EJECUTAR RÉPLICAS
# ============================================================

def ejecutar_replicas(n_replicas=5, n_clientes=100, semilla_base=2026):
    """Ejecuta réplicas nuevas, cada una con el estado del simulador reiniciado.

    El índice r empieza en 0; cada semilla es semilla_base + r * 1009.
    """

    filas = []

    simulaciones = {}


    for r in range(n_replicas):

        semilla = semilla_base + r * 1009


        # ----------------------------------------------------
        # EJECUTAR UNA SIMULACIÓN
        # ----------------------------------------------------

        df = simular_gasolinera(n_clientes, semilla)


        resumen = resumen_simulacion(df)


        filas.append({

            "replica": r + 1,

            "semilla": semilla,

            "n_clientes": n_clientes,

            "espera_media": resumen["espera_media"],

            "espera_maxima": resumen["espera_maxima"],

            "cola_maxima": resumen["cola_maxima"],

            "tiempo_medio_sistema": resumen["tiempo_medio_sistema"],

            "utilizacion_observada": resumen["utilizacion_observada"],

            "horizonte": resumen["horizonte"]
        })


        simulaciones[r + 1] = df


    resultados = pd.DataFrame(filas)


    return resultados, simulaciones
