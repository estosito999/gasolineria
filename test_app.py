"""Pruebas de integración de las seis pestañas de la aplicación Streamlit.

Ejecutar desde gasolineria con: python -m unittest test_app -v
"""

import json
from pathlib import Path
import subprocess
import sys
import textwrap
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from estadistica import intervalo_confianza


DIRECTORIO = Path(__file__).resolve().parent
COLUMNA_ESPERA = "Tiempo medio de espera en cola (min)"
MEDIAS_OFICIALES = [67.15, 71.72, 52.47, 68.19, 46.05]
SEMILLAS_OFICIALES = [2026, 3035, 4044, 5053, 6062]


class ImportacionModulosTests(unittest.TestCase):
    def test_importar_las_seis_instrucciones_no_dibuja_ui(self):
        # Otro proceso garantiza una importación inicial, sin módulos en caché.
        programa = textwrap.dedent("""
            import importlib
            from contextlib import ExitStack
            from unittest.mock import patch
            import streamlit as st

            comandos_ui = (
                "set_page_config", "title", "header", "subheader", "write",
                "caption", "markdown", "metric", "dataframe", "table",
                "plotly_chart", "tabs", "columns", "expander", "slider",
                "number_input", "info", "warning", "success", "latex",
            )
            with ExitStack() as pila:
                for comando in comandos_ui:
                    pila.enter_context(patch.object(
                        st, comando,
                        side_effect=AssertionError("UI ejecutada durante importación: " + comando),
                    ))
                for numero in range(1, 7):
                    modulo = importlib.import_module("Instruccion" + str(numero))
                    assert callable(getattr(modulo, "mostrar_instruccion" + str(numero)))
            print("IMPORTACIONES_OK")
        """)
        proceso = subprocess.run(
            [sys.executable, "-c", programa],
            cwd=DIRECTORIO,
            capture_output=True,
            text=True,
            timeout=30,
        )

        self.assertEqual(proceso.returncode, 0, proceso.stdout + proceso.stderr)
        self.assertIn("IMPORTACIONES_OK", proceso.stdout)


class AppIntegradaTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(DIRECTORIO / "app.py", default_timeout=30).run()
        self.comprobar_sin_errores()

    def comprobar_sin_errores(self):
        self.assertFalse(self.app.exception, [error.message for error in self.app.exception])

    def tabla_con_columnas(self, pestana, columnas, filas=None):
        candidatas = [
            tabla.value for tabla in pestana.dataframe
            if set(columnas).issubset(tabla.value.columns)
            and (filas is None or len(tabla.value) == filas)
        ]
        self.assertEqual(len(candidatas), 1, f"Tabla ausente o ambigua: {columnas}")
        return candidatas[0]

    def metricas(self, pestana):
        return {metrica.label: metrica.value for metrica in pestana.metric}

    def resultados_visibles(self):
        replicas = self.tabla_con_columnas(
            self.app.tabs[4],
            ["Réplica", "Semilla utilizada", "Número de clientes", COLUMNA_ESPERA],
        )
        medias = self.tabla_con_columnas(self.app.tabs[5], ["Réplica", COLUMNA_ESPERA])
        return replicas.copy(deep=True), medias.copy(deep=True), self.metricas(self.app.tabs[5])

    def test_seis_pestanas_contienen_encabezados_y_datos_reales(self):
        encabezados = [
            "Análisis de los datos observados",
            "Generación de variables aleatorias",
            "Análisis de distribución",
            "Construcción del modelo",
            "Ejecución de réplicas",
            "Análisis estadístico de los resultados",
        ]
        self.assertEqual(len(self.app.tabs), 6)
        for numero, (pestana, encabezado) in enumerate(zip(self.app.tabs, encabezados), start=1):
            with self.subTest(instruccion=numero):
                self.assertIn(f"Instrucción {numero}", pestana.label)
                self.assertEqual(len(pestana.header), 1)
                self.assertIn(encabezado, pestana.header[0].value)
                self.assertTrue(pestana.subheader)
                self.assertTrue(any(not tabla.value.empty for tabla in pestana.dataframe))

        datos_llegadas = self.tabla_con_columnas(
            self.app.tabs[0], ["Tiempo de interarribo (min)", "Frecuencia"]
        )
        self.assertEqual(datos_llegadas["Frecuencia"].tolist(), [10, 15, 15, 7, 3])
        generacion = self.tabla_con_columnas(
            self.app.tabs[1], ["Vehículo", "U interarribo", "Interarribo (min)", "U servicio"]
        )
        self.assertEqual(generacion["Vehículo"].tolist(), [1, 2, 3, 4, 5])
        self.assertTrue(generacion["U interarribo"].between(0, 1, inclusive="left").all())
        registro = self.tabla_con_columnas(
            self.app.tabs[3], ["Vehículo", "Llegada", "Inicio servicio", "Espera en cola"], filas=100
        )
        self.assertEqual(registro["Vehículo"].tolist(), list(range(1, 101)))
        np.testing.assert_array_equal(
            registro["Espera en cola"], registro["Inicio servicio"] - registro["Llegada"]
        )

    def test_instrucciones_cinco_y_seis_comparten_las_medias_oficiales_y_el_ic(self):
        replicas, medias, metricas = self.resultados_visibles()

        self.assertEqual(replicas["Réplica"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(replicas["Semilla utilizada"].tolist(), SEMILLAS_OFICIALES)
        self.assertEqual(replicas["Número de clientes"].tolist(), [100] * 5)
        np.testing.assert_allclose(replicas[COLUMNA_ESPERA], MEDIAS_OFICIALES)
        pd.testing.assert_frame_equal(
            replicas[["Réplica", COLUMNA_ESPERA]], medias
        )

        intervalo = intervalo_confianza(medias[COLUMNA_ESPERA].to_numpy(), confianza=0.95)
        self.assertAlmostEqual(intervalo["media"], 61.116, places=10)
        self.assertEqual(metricas["⏳ Media global de espera en cola"], f"{intervalo['media']:.2f} min")
        self.assertEqual(metricas["Límite inferior"], f"{intervalo['li']:.2f} min")
        self.assertEqual(metricas["Límite superior"], f"{intervalo['ls']:.2f} min")
        self.assertEqual(metricas["Semiamplitud"], f"{intervalo['semiancho']:.2f} min")

    def test_reloj_y_vehiculo_cien_hacen_rerun_sin_cambiar_las_replicas(self):
        replicas, medias, metricas = self.resultados_visibles()
        reloj = self.app.tabs[2].slider[0]
        horizonte = int(reloj.proto.max)
        self.assertEqual(reloj.label, "Tiempo")
        self.assertEqual(self.app.tabs[2].number_input[0].label, "Vehículo")
        self.assertEqual(self.app.tabs[2].number_input[0].proto.max, 100)

        reloj.set_value(horizonte // 2)
        self.app.run()
        self.comprobar_sin_errores()
        self.assertEqual(self.app.tabs[2].slider[0].value, horizonte // 2)
        replicas_intermedias, medias_intermedias, metricas_intermedias = self.resultados_visibles()
        pd.testing.assert_frame_equal(replicas, replicas_intermedias)
        pd.testing.assert_frame_equal(medias, medias_intermedias)
        self.assertEqual(metricas, metricas_intermedias)

        self.app.tabs[2].slider[0].set_value(horizonte)
        self.app.tabs[2].number_input[0].set_value(100)
        self.app.run()
        self.comprobar_sin_errores()
        self.assertEqual(self.app.tabs[2].number_input[0].value, 100)
        metricas_reloj = self.metricas(self.app.tabs[2])
        self.assertEqual(metricas_reloj["Finalizados"], "100")
        self.assertEqual(metricas_reloj["Esperando"], "0")
        self.assertEqual(metricas_reloj["Surtidor"], "LIBRE")

        registro = self.tabla_con_columnas(self.app.tabs[2], ["vehiculo", "espera_cola"], filas=100)
        ultimo = registro.iloc[-1]
        inspeccion = json.loads(self.app.tabs[2].json[0].value)
        self.assertEqual(inspeccion, {
            "llegada": int(ultimo["llegada"]),
            "servicio": int(ultimo["servicio"]),
            "inicio": int(ultimo["inicio_servicio"]),
            "salida": int(ultimo["salida"]),
            "espera": int(ultimo["espera_cola"]),
        })
        replicas_finales, medias_finales, metricas_finales = self.resultados_visibles()
        pd.testing.assert_frame_equal(replicas, replicas_finales)
        pd.testing.assert_frame_equal(medias, medias_finales)
        self.assertEqual(metricas, metricas_finales)


if __name__ == "__main__":
    unittest.main()
