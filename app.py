import streamlit as st

from Instruccion1 import mostrar_instruccion1
from Instruccion2 import mostrar_instruccion2
from Instruccion3 import mostrar_instruccion3
from Instruccion4 import mostrar_instruccion4
from Instruccion5 import mostrar_instruccion5
from Instruccion6 import mostrar_instruccion6
from datos import obtener_distribuciones
from simulador import resumen_simulacion, simular_gasolinera

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Gasolinera",
    page_icon="⛽",
    layout="wide"
)


# ============================================================
# PARÁMETROS FIJOS DEL EJERCICIO
# ============================================================

n_clientes = 100
semilla = 2026
n_replicas = 5
confianza = 0.95
precision = 0.10

# ============================================================
# TÍTULO
# ============================================================

st.title("⛽ Simulación interactiva de una gasolinera")

st.caption(
    "FIFO · 1 surtidor · Transformada inversa · "
    "Eventos discretos · Réplicas"
)


# ============================================================
# DATOS
# ============================================================

tabla_llegadas, tabla_servicio = obtener_distribuciones()

# ============================================================
# SIMULACIÓN PRINCIPAL
# ============================================================

simulacion = simular_gasolinera(
    int(n_clientes),
    int(semilla)
)


resumen = resumen_simulacion(
    simulacion
)


# ============================================================
# PESTAÑAS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Instrucción 1",
    "🎲 Instrucción 2",
    "📐 Instrucción 3",
    "🧪 Instrucción 4",
    "🧪 Instrucción 5",
    "📈 Instrucción 6"
])

with tab1:
    mostrar_instruccion1(tabla_llegadas, tabla_servicio)

with tab2:
    mostrar_instruccion2(tabla_llegadas, tabla_servicio, n_clientes, semilla)

with tab3:
    mostrar_instruccion3(simulacion, resumen, n_clientes)

with tab4:
    mostrar_instruccion4(simulacion)

with tab5:
    resultados_replicas, simulaciones_replicas = mostrar_instruccion5(
        n_replicas, n_clientes, semilla
    )

with tab6:
    mostrar_instruccion6(resultados_replicas, confianza, precision)
