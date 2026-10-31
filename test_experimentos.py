"""Regresiones de las réplicas independientes del experimento de gasolinera."""

import unittest

import numpy as np
import pandas as pd

from experimentos import ejecutar_replicas
from simulador import simular_gasolinera


class EjecutarReplicasTests(unittest.TestCase):
    def test_lote_oficial_cinco_replicas_de_cien_clientes(self):
        resultados, simulaciones = ejecutar_replicas()

        self.assertEqual(resultados["replica"].tolist(), [1, 2, 3, 4, 5])
        self.assertEqual(resultados["semilla"].tolist(), [2026, 3035, 4044, 5053, 6062])
        self.assertEqual(resultados["n_clientes"].tolist(), [100] * 5)
        np.testing.assert_allclose(resultados["espera_media"], [67.15, 71.72, 52.47, 68.19, 46.05])
        self.assertEqual(set(simulaciones), {1, 2, 3, 4, 5})

        for registro in resultados.itertuples(index=False):
            with self.subTest(replica=registro.replica):
                espera = simulaciones[registro.replica]["espera_cola"]
                self.assertEqual(len(espera), 100)
                self.assertTrue((espera == 0).any())
                self.assertAlmostEqual(registro.espera_media, espera.mean())
                self.assertAlmostEqual(registro.espera_media, espera.sum() / 100)

    def test_mismo_lote_es_reproducible(self):
        resultados_1, simulaciones_1 = ejecutar_replicas()
        resultados_2, simulaciones_2 = ejecutar_replicas()

        pd.testing.assert_frame_equal(resultados_1, resultados_2)
        self.assertEqual(simulaciones_1.keys(), simulaciones_2.keys())
        for replica in simulaciones_1:
            with self.subTest(replica=replica):
                pd.testing.assert_frame_equal(simulaciones_1[replica], simulaciones_2[replica])

    def test_cada_replica_coincide_con_una_corrida_aislada_y_arranca_vacia(self):
        resultados, simulaciones = ejecutar_replicas()

        for registro in resultados.itertuples(index=False):
            with self.subTest(replica=registro.replica):
                simulacion = simulaciones[registro.replica]
                aislada = simular_gasolinera(n_clientes=100, semilla=registro.semilla)
                pd.testing.assert_frame_equal(simulacion, aislada)

                primero = simulacion.iloc[0]
                self.assertEqual(primero["vehiculo"], 1)
                self.assertEqual(primero["cola_al_llegar"], 0)
                self.assertEqual(primero["espera_cola"], 0)
                self.assertEqual(primero["inicio_servicio"], primero["llegada"])

    def test_api_permite_otro_tamano_de_experimento_y_semilla_base(self):
        resultados, simulaciones = ejecutar_replicas(n_replicas=2, n_clientes=12, semilla_base=7)

        self.assertEqual(resultados["replica"].tolist(), [1, 2])
        self.assertEqual(resultados["semilla"].tolist(), [7, 1016])
        self.assertEqual(resultados["n_clientes"].tolist(), [12, 12])
        self.assertEqual(set(simulaciones), {1, 2})
        for registro in resultados.itertuples(index=False):
            with self.subTest(replica=registro.replica):
                simulacion = simulaciones[registro.replica]
                self.assertEqual(len(simulacion), 12)
                self.assertAlmostEqual(registro.espera_media, simulacion["espera_cola"].mean())


if __name__ == "__main__":
    unittest.main()
