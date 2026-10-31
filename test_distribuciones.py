"""Regresiones del ajuste de Poisson sobre frecuencias resumidas.

Ejecutar desde gasolineria con: python -m unittest test_distribuciones -v
"""

import math
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import distribuciones
from datos import LLEGADAS, SERVICIO
from distribuciones import agrupar_clases, ajuste_poisson
from simulador import simular_gasolinera


class AjustePoissonTests(unittest.TestCase):
    def test_referencias_de_las_frecuencias_proporcionadas(self):
        referencias = [
            (LLEGADAS, 2.56, 4.472487793293231, 3, 0.2147574806489818, False),
            (SERVICIO, 4.0, 20.880801447776353, 4, 0.0003343742409949921, True),
        ]
        for datos, media, estadistico, gl, pvalor, rechazar in referencias:
            with self.subTest(media=media):
                resultado = ajuste_poisson(datos)
                tabla = resultado["tabla"]

                self.assertTrue(resultado["valido"])
                self.assertAlmostEqual(resultado["lambda_estimado"], media)
                self.assertAlmostEqual(resultado["chi2"], estadistico, places=10)
                self.assertEqual(resultado["gl"], gl)
                self.assertEqual(resultado["gl"], len(tabla) - 1 - 1)
                self.assertAlmostEqual(resultado["pvalor"], pvalor, places=12)
                self.assertEqual(resultado["alpha"], 0.05)
                self.assertIs(type(resultado["rechazar"]), bool)
                self.assertIs(resultado["rechazar"], rechazar)
                self.assertEqual(tabla["observada"].sum(), datos["frecuencia"].sum())
                self.assertAlmostEqual(tabla["esperada"].sum(), datos["frecuencia"].sum())
                self.assertTrue((tabla["esperada"] >= 5).all())

                contribuciones = (tabla["observada"] - tabla["esperada"]) ** 2 / tabla["esperada"]
                np.testing.assert_allclose(tabla["contribucion"], contribuciones)
                self.assertAlmostEqual(tabla["contribucion"].sum(), resultado["chi2"])

    def test_soporte_completo_incluye_ceros_categorias_faltantes_y_cola(self):
        datos = pd.DataFrame({"tiempo": [1, 3, 5], "frecuencia": [25, 50, 25]})

        with patch.object(distribuciones, "agrupar_clases", wraps=agrupar_clases) as agrupar:
            resultado = ajuste_poisson(datos)

        observados, esperados, etiquetas = agrupar.call_args.args[:3]
        self.assertEqual(list(observados), [0, 25, 0, 50, 0, 25, 0])
        self.assertEqual(list(etiquetas), ["0", "1", "2", "3", "4", "5", ">=6"])
        probabilidades = [math.exp(-3) * 3 ** k / math.factorial(k) for k in range(6)]
        esperados_teoricos = [100 * p for p in probabilidades]
        esperados_teoricos.append(100 * (1 - sum(probabilidades)))
        np.testing.assert_allclose(esperados, esperados_teoricos)
        self.assertEqual(sum(observados), 100)
        self.assertAlmostEqual(sum(esperados), 100)
        self.assertEqual(resultado["tabla"]["observada"].sum(), 100)
        self.assertAlmostEqual(resultado["tabla"]["esperada"].sum(), 100)

    def test_tiempos_duplicados_acumulan_frecuencias_sin_modificar_datos(self):
        datos = pd.DataFrame({
            "tiempo": [1, 1, 2, 2, 3, 4, 5],
            "frecuencia": [4, 6, 5, 10, 15, 7, 3],
        })
        copia = datos.copy(deep=True)
        resultado = ajuste_poisson(datos)
        referencia = ajuste_poisson(LLEGADAS)

        pd.testing.assert_frame_equal(datos, copia)
        pd.testing.assert_frame_equal(resultado["tabla"], referencia["tabla"])
        for campo in ("lambda_estimado", "chi2", "gl", "pvalor", "rechazar"):
            self.assertEqual(resultado[campo], referencia[campo])

    def test_alpha_controla_la_decision_incluida_la_igualdad(self):
        resultado = ajuste_poisson(LLEGADAS, alpha=0.3)
        self.assertIs(resultado["rechazar"], True)
        self.assertEqual(resultado["alpha"], 0.3)

        igualdad = ajuste_poisson(LLEGADAS, alpha=resultado["pvalor"])
        self.assertIs(igualdad["rechazar"], False)

    def test_condiciones_insuficientes_no_producen_decision_inferencial(self):
        casos = [
            pd.DataFrame({"tiempo": [0, 1, 2], "frecuencia": [1, 1, 1]}),
            pd.DataFrame({"tiempo": [0], "frecuencia": [50]}),
        ]
        for datos in casos:
            with self.subTest(datos=datos.to_dict("list")):
                resultado = ajuste_poisson(datos)
                self.assertIs(resultado["valido"], False)
                self.assertTrue(resultado["motivo"])
                for campo in ("lambda_estimado", "alpha", "gl", "tabla"):
                    self.assertIn(campo, resultado)
                self.assertEqual(resultado["alpha"], 0.05)
                self.assertIsNone(resultado["pvalor"])
                self.assertIsNone(resultado["rechazar"])
                self.assertTrue(
                    resultado["gl"] <= 0
                    or (resultado["tabla"]["esperada"] < 5).any()
                    or resultado["lambda_estimado"] == 0
                )
                if resultado["gl"] <= 0 or (resultado["tabla"]["esperada"] <= 0).any():
                    self.assertIsNone(resultado["chi2"])

    def test_rechaza_tiempos_fuera_del_soporte_poisson(self):
        for tiempo in (-1, 0.5, math.nan, math.inf, "invalido"):
            with self.subTest(tiempo=tiempo):
                datos = pd.DataFrame({"tiempo": [tiempo], "frecuencia": [10]})
                with self.assertRaises(ValueError):
                    ajuste_poisson(datos)

    def test_rechaza_frecuencias_invalidas(self):
        for frecuencia in (-1, 0.5, math.nan, math.inf, "invalida"):
            with self.subTest(frecuencia=frecuencia):
                datos = pd.DataFrame({"tiempo": [1], "frecuencia": [frecuencia]})
                with self.assertRaises(ValueError):
                    ajuste_poisson(datos)

    def test_rechaza_datos_vacios_y_total_cero(self):
        casos = [
            pd.DataFrame(columns=["tiempo", "frecuencia"]),
            pd.DataFrame({"tiempo": [0, 1], "frecuencia": [0, 0]}),
        ]
        for datos in casos:
            with self.subTest(datos=datos.to_dict("list")):
                with self.assertRaises(ValueError):
                    ajuste_poisson(datos)

    def test_rechaza_alpha_fuera_de_intervalo(self):
        for alpha in (-0.1, 0, 1, 1.1, math.nan, math.inf):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    ajuste_poisson(LLEGADAS, alpha=alpha)

    def test_ajuste_no_cambia_el_generador_empirico_de_la_simulacion(self):
        llegadas_originales = LLEGADAS.copy(deep=True)
        servicio_original = SERVICIO.copy(deep=True)
        antes = simular_gasolinera(n_clientes=8, semilla=2026)

        ajuste_poisson(LLEGADAS)
        ajuste_poisson(SERVICIO)

        despues = simular_gasolinera(n_clientes=8, semilla=2026)
        pd.testing.assert_frame_equal(antes, despues)
        pd.testing.assert_frame_equal(LLEGADAS, llegadas_originales)
        pd.testing.assert_frame_equal(SERVICIO, servicio_original)
        self.assertEqual(despues["entre_llegadas"].tolist(), [1, 3, 2, 2, 2, 3, 4, 1])
        self.assertEqual(despues["servicio"].tolist(), [4, 3, 6, 6, 4, 5, 4, 5])


class AgrupacionClasesTests(unittest.TestCase):
    def test_agrupa_y_fusiona_sobrante_conservando_totales(self):
        observados = [5, 10, 15, 7]
        esperados = [3.0, 4.0, 5.0, 2.0]
        etiquetas = ["0", "1", "2", ">=3"]
        grupos = agrupar_clases(observados, esperados, etiquetas)

        self.assertEqual(grupos, [[15, 7.0, "0 + 1"], [22, 7.0, "2 + >=3"]])
        self.assertEqual(sum(grupo[0] for grupo in grupos), sum(observados))
        self.assertAlmostEqual(sum(grupo[1] for grupo in grupos), sum(esperados))
        self.assertTrue(all(grupo[1] >= 5 for grupo in grupos))

    def test_muestra_insuficiente_conserva_un_unico_grupo(self):
        self.assertEqual(agrupar_clases([1, 2], [1.0, 2.0], ["0", ">=1"]),
                         [[3, 3.0, "0 + >=1"]])

    def test_rechaza_longitudes_distintas(self):
        casos = [([1], [1, 2], ["0"]), ([1, 2], [1], ["0"]), ([1], [1], ["0", "1"])]
        for observados, esperados, etiquetas in casos:
            with self.subTest(observados=observados, esperados=esperados, etiquetas=etiquetas):
                with self.assertRaises(ValueError):
                    agrupar_clases(observados, esperados, etiquetas)


if __name__ == "__main__":
    unittest.main()
