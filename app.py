import math

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t

from datos import LLEGADAS
from datos import SERVICIO
from datos import obtener_distribuciones
from datos import resumen_datos

from generador import transformar_uniforme

from simulador import (
    simular_gasolinera,
    resumen_simulacion,
    estado_en_tiempo,
    longitud_cola_en_tiempo,
)

from experimentos import ejecutar_replicas

from estadistica import (
    intervalo_confianza,
    replicas_requeridas,
    resumen_estadistico,
)

from distribuciones import ajuste_poisson

from visualizacion import (
    grafico_timeline,
    grafico_replicas,
)

from exportacion import *


# ============================================================
# CONFIGURACION DE LA PAGINA
# ============================================================

st.set_page_config(
    page_title="Simulacion · Gasolinera",
    layout="wide"
)


# ============================================================
# ESTILOS CSS GLOBALES
# ============================================================

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Contenedor de metricas estilizado */
    [data-testid="metric-container"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
    }

    /* Encabezados de seccion con icono */
    .section-header {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 14px;
        margin-bottom: 8px;
    }
    .section-header svg { flex-shrink: 0; }

    /* Bloque de consigna / contexto institucional */
    .instr-block {
        background: #f1f5f9;
        border-left: 4px solid #64748b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 16px;
        font-size: 0.92rem;
        color: #334155;
        line-height: 1.5;
    }

    /* Tarjetas de hipotesis H0 y H1 */
    .h0-block {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 4px solid #16a34a;
        border-radius: 6px;
        padding: 14px 16px;
        height: 100%;
    }
    .h1-block {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-left: 4px solid #dc2626;
        border-radius: 6px;
        padding: 14px 16px;
        height: 100%;
    }

    /* Bloque para preguntas tecnicas */
    .q-block {
        background: #f0fdfa;
        border-left: 4px solid #0d9488;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .q-block .q-label {
        font-size: 0.76rem;
        font-weight: 700;
        color: #0f766e;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .q-block .q-text {
        font-size: 0.96rem;
        color: #134e4a;
        font-weight: 600;
    }

    /* Bloque para pensamiento critico */
    .crit-block {
        background: #faf5ff;
        border: 1px solid #f3e8ff;
        border-left: 4px solid #9333ea;
        border-radius: 6px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .crit-block .crit-q {
        font-size: 0.76rem;
        font-weight: 700;
        color: #7e22ce;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .crit-block .crit-title {
        font-size: 0.97rem;
        font-weight: 700;
        color: #3b0764;
        margin-bottom: 8px;
    }
    .crit-block .crit-a {
        font-size: 0.92rem;
        color: #1e1b4b;
        line-height: 1.6;
    }

    /* Bloque de interpretacion destacada */
    .interp-block {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        border-radius: 6px;
        padding: 12px 16px;
        margin-top: 10px;
        margin-bottom: 10px;
        font-size: 0.92rem;
        color: #1e3a8a;
        line-height: 1.55;
    }

    /* Pestañas limpias */
    button[data-baseweb="tab"] {
        font-size: 0.86rem !important;
        padding: 8px 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# ICONOS SVG INLINE (SIN EMOJIS)
# ============================================================

def svg(path_d, size=18, color="#475569"):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
        f'<path d="{path_d}"/></svg>'
    )

ICO_CHART   = "M3 3v18h18M9 17V9m4 8V5m4 12v-5"
ICO_SHUFFLE = "M16 3h5v5M4 20 21 3M21 16v5h-5M15 15l5.1 5.1M4 4l5 5"
ICO_DIST    = "M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18"
ICO_SIM     = "M12 2a10 10 0 1 0 0 20A10 10 0 0 0 12 2zm0 6v6l4 2"
ICO_REP     = "M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2M12 7a4 4 0 1 0 0 8 4 4 0 0 0 0-8z"
ICO_BOOK    = "M4 19.5A2.5 2.5 0 0 1 6.5 17H20M4 19.5A2.5 2.5 0 0 0 6.5 22H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15z"
ICO_TECH    = "M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 0 2 2v4M9 3v18m0 0h10a2 2 0 0 0 2-2V9M9 21H5a2 2 0 0 0-2-2V9m0 0h18"
ICO_BRAIN   = "M12 2a7 7 0 0 1 7 7c0 4-3 6-5 8H10c-2-2-5-4-5-8a7 7 0 0 1 7-7zm-2 17h4m-2 2v1"
ICO_CHECK   = "M20 6 9 17l-5-5"
ICO_CALC    = "M4 2h16a2 2 0 0 1 2 2v16a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2zm4 8h8M8 14h8M8 6h2m4 0h2"


def section_title(icon_path, text, color="#0f172a", size=20):
    st.markdown(
        f'<div class="section-header">'
        f'{svg(icon_path, size=size, color=color)}'
        f'<span style="font-size:1.05rem;font-weight:700;color:{color};">{text}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def instr_context(texto):
    st.markdown(
        f'<div class="instr-block">{texto}</div>',
        unsafe_allow_html=True,
    )


def interpretacion(texto):
    st.markdown(
        f'<div class="interp-block">{texto}</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# DATOS BASE Y CALCULOS PREVIOS
# ============================================================

tabla_llegadas, tabla_servicio = obtener_distribuciones()
datos_base = resumen_datos()


# ============================================================
# BARRA LATERAL (SIDEBAR)
# ============================================================

with st.sidebar:
    st.markdown(
        '<p style="font-size:1.05rem;font-weight:700;margin-bottom:2px;color:#0f172a;">'
        'Parametros de simulacion</p>',
        unsafe_allow_html=True,
    )
    st.caption("Escenario 2 · Gasolinera (FIFO · 1 surtidor)")
    st.divider()

    n_clientes = st.number_input(
        "Vehiculos por replica",
        min_value=10, max_value=1000, value=100, step=10,
        help="Cantidad de vehiculos simulados en cada corrida."
    )

    semilla = st.number_input(
        "Semilla aleatoria",
        min_value=0, value=2026, step=1,
        help="Semilla base pseudoaleatoria para reproducibilidad."
    )

    n_replicas = st.number_input(
        "Numero de replicas",
        min_value=2, max_value=100, value=5, step=1,
        help="Cantidad de corridas independientes ejecutadas."
    )

    confianza = st.slider(
        "Nivel de confianza (1 - alpha)",
        min_value=0.80, max_value=0.99, value=0.95, step=0.01,
        format="%.2f",
        help="Nivel de confianza para el intervalo t de Student (default 95%)."
    )

    precision = st.slider(
        "Error relativo maximo deseado",
        min_value=0.01, max_value=0.30, value=0.10, step=0.01,
        format="%.2f",
        help="Precision relativa deseada para estimar replicas necesarias (10%)."
    )

    st.divider()
    st.caption(
        "Condiciones de simulación:\n"
        "• Sistema: Una cola y un servidor\n"
        "• Disciplina: FIFO\n"
        "• Unidad de tiempo: Minutos\n"
        "• Estado inicial: Servidor libre, cola vacia"
    )


# ============================================================
# ENCABEZADO PRINCIPAL
# ============================================================

st.markdown(
    '<h1 style="font-size:1.75rem;font-weight:800;margin-bottom:2px;color:#0f172a;">'
    'Simulacion de Gasolinera — Escenario 2</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#64748b;margin-top:0;font-size:0.92rem;">'
    'FIFO · 1 surtidor · Transformada inversa · Eventos discretos · Replicas independientes</p>',
    unsafe_allow_html=True,
)


# ============================================================
# EJECUCION DE SIMULACION PRINCIPAL Y REPLICAS
# ============================================================

simulacion = simular_gasolinera(int(n_clientes), int(semilla))
resumen = resumen_simulacion(simulacion)
resultados_rep, simulaciones_rep = ejecutar_replicas(
    int(n_replicas), int(n_clientes), int(semilla)
)


# ============================================================
# PESTANAS DEL SISTEMA (6 INSTRUCCIONES + 2 ANALISIS)
# ============================================================

(
    tab1,
    tab2,
    tab3,
    tab4,
    tab5,
    tab6,
    tab7,
    tab8,
) = st.tabs([
    "1. Datos observados",
    "2. Transformada inversa",
    "3. Analisis distribucion",
    "4. Modelo",
    "5. Replicas",
    "6. Conclusiones",
    "Preg. tecnicas",
    "Pens. critico",
])


# ============================================================
# TAB 1: INSTRUCCION 1 — DATOS OBSERVADOS
# ============================================================

with tab1:

    section_title(ICO_CHART, "Instruccion 1 — Analisis de datos observados")

    instr_context(
        "Analizar los datos observados. Calcular las medias y frecuencias relativas "
        "de los tiempos entre llegadas y de servicio."
    )

    st.divider()

    # Tablas de frecuencia observada con frecuencias relativas y acumuladas
    col_t1, col_t2 = st.columns(2)

    with col_t1:
        section_title(ICO_CHART, "A. Tiempos entre llegadas", color="#0284c7")

        df_llegadas_tab = tabla_llegadas[
            ["tiempo", "frecuencia", "probabilidad", "acumulada"]
        ].copy()
        df_llegadas_tab.columns = [
            "Tiempo (min)",
            "Frecuencia (f_i)",
            "Frecuencia relativa (p_i)",
            "Frecuencia acumulada (F_i)",
        ]
        df_llegadas_tab["Frecuencia relativa (p_i)"] = df_llegadas_tab[
            "Frecuencia relativa (p_i)"
        ].map("{:.4f}".format)
        df_llegadas_tab["Frecuencia acumulada (F_i)"] = df_llegadas_tab[
            "Frecuencia acumulada (F_i)"
        ].map("{:.4f}".format)

        total_n_ll = int(LLEGADAS["frecuencia"].sum())
        df_llegadas_full = pd.concat([
            df_llegadas_tab,
            pd.DataFrame([{
                "Tiempo (min)": "Total",
                "Frecuencia (f_i)": total_n_ll,
                "Frecuencia relativa (p_i)": "1.0000",
                "Frecuencia acumulada (F_i)": "—",
            }])
        ], ignore_index=True)

        st.dataframe(df_llegadas_full, use_container_width=True, hide_index=True)

    with col_t2:
        section_title(ICO_CHART, "B. Tiempos de servicio", color="#0284c7")

        df_servicio_tab = tabla_servicio[
            ["tiempo", "frecuencia", "probabilidad", "acumulada"]
        ].copy()
        df_servicio_tab.columns = [
            "Tiempo (min)",
            "Frecuencia (f_i)",
            "Frecuencia relativa (p_i)",
            "Frecuencia acumulada (F_i)",
        ]
        df_servicio_tab["Frecuencia relativa (p_i)"] = df_servicio_tab[
            "Frecuencia relativa (p_i)"
        ].map("{:.4f}".format)
        df_servicio_tab["Frecuencia acumulada (F_i)"] = df_servicio_tab[
            "Frecuencia acumulada (F_i)"
        ].map("{:.4f}".format)

        total_n_sv = int(SERVICIO["frecuencia"].sum())
        df_servicio_full = pd.concat([
            df_servicio_tab,
            pd.DataFrame([{
                "Tiempo (min)": "Total",
                "Frecuencia (f_i)": total_n_sv,
                "Frecuencia relativa (p_i)": "1.0000",
                "Frecuencia acumulada (F_i)": "—",
            }])
        ], ignore_index=True)

        st.dataframe(df_servicio_full, use_container_width=True, hide_index=True)

    st.divider()

    # Medias de llegada y de servicio con desarrollo matematico paso a paso
    section_title(ICO_CALC, "Calculo de medias muestrales ponderadas", color="#0284c7")

    st.write(
        "La media de cada variable discreta se calcula como el promedio ponderado "
        "de los valores observados por sus frecuencias absolutas correspondientes:"
    )

    st.latex(r"\bar{X} = \frac{\sum_{i=1}^{k} x_i \cdot f_i}{\sum_{i=1}^{k} f_i}")

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("**Media del tiempo entre llegadas**")
        desarrollo_ll = " + ".join([
            f"({t} \\cdot {f})" for t, f in zip(LLEGADAS["tiempo"], LLEGADAS["frecuencia"])
        ])
        suma_prod_ll = int(sum(t * f for t, f in zip(LLEGADAS["tiempo"], LLEGADAS["frecuencia"])))

        st.latex(
            rf"\bar{{X}}_{{llegada}} = \frac{{{desarrollo_ll}}}{{{total_n_ll}}} = "
            rf"\frac{{{suma_prod_ll}}}{{{total_n_ll}}} = {datos_base['media_entre_llegadas']:.2f}\text{{ min}}"
        )
        st.metric(
            "Media de tiempo entre llegadas",
            f"{datos_base['media_entre_llegadas']:.2f} min"
        )

    with col_m2:
        st.markdown("**Media del tiempo de servicio**")
        desarrollo_sv = " + ".join([
            f"({t} \\cdot {f})" for t, f in zip(SERVICIO["tiempo"], SERVICIO["frecuencia"])
        ])
        suma_prod_sv = int(sum(t * f for t, f in zip(SERVICIO["tiempo"], SERVICIO["frecuencia"])))

        st.latex(
            rf"\bar{{X}}_{{servicio}} = \frac{{{desarrollo_sv}}}{{{total_n_sv}}} = "
            rf"\frac{{{suma_prod_sv}}}{{{total_n_sv}}} = {datos_base['media_servicio']:.2f}\text{{ min}}"
        )
        st.metric(
            "Media de tiempo de servicio",
            f"{datos_base['media_servicio']:.2f} min"
        )

    st.divider()

    # Carga del sistema rho
    section_title(ICO_CHECK, "Tasas del sistema y carga de utilizacion (rho)", color="#0284c7")

    c_r1, c_r2, c_r3, c_r4 = st.columns(4)
    c_r1.metric("Media entre llegadas", f"{datos_base['media_entre_llegadas']:.2f} min")
    c_r2.metric("Media de servicio", f"{datos_base['media_servicio']:.2f} min")
    c_r3.metric("Tasa de llegada (lambda)", f"{datos_base['lambda_hora']:.2f} veh/h")
    c_r4.metric("Capacidad de atencion (mu)", f"{datos_base['mu_hora']:.2f} veh/h")

    st.latex(
        r"\lambda = \frac{1}{\bar{X}_{llegada}} = "
        rf"\frac{{1}}{{{datos_base['media_entre_llegadas']:.2f}}} = {datos_base['lambda_min']:.4f}\text{{ veh/min}} "
        rf"\quad ({datos_base['lambda_hora']:.2f}\text{{ veh/h}})"
    )

    st.latex(
        r"\mu = \frac{1}{\bar{X}_{servicio}} = "
        rf"\frac{{1}}{{{datos_base['media_servicio']:.2f}}} = {datos_base['mu_min']:.4f}\text{{ veh/min}} "
        rf"\quad ({datos_base['mu_hora']:.2f}\text{{ veh/h}})"
    )

    st.latex(
        r"\rho = \frac{\lambda}{\mu} = \frac{\bar{X}_{servicio}}{\bar{X}_{llegada}} = "
        rf"\frac{{{datos_base['media_servicio']:.2f}}}{{{datos_base['media_entre_llegadas']:.2f}}} = "
        rf"{datos_base['rho']:.4f}"
    )

    st.divider()

    # Interpretacion de los datos observados
    section_title(ICO_BOOK, "Interpretacion de los datos observados", color="#0284c7")

    interpretacion(
        f"<b>1. Frecuencias relativas:</b> En los tiempos entre llegadas, los tiempos de 2 y 3 minutos "
        f"son los mas frecuentes (30 % cada uno), totalizando el 60 % de las observaciones. "
        f"En el servicio, el tiempo modal es 4 minutos (40 % de los casos), con una distribucion simetrica.<br>"
        f"<b>2. Comparativa entre medias:</b> El tiempo medio entre llegadas es de <b>{datos_base['media_entre_llegadas']:.2f} minutos</b>, "
        f"mientras que el tiempo medio de servicio es de <b>{datos_base['media_servicio']:.2f} minutos</b>. "
        f"Cada vehiculo tarda en promedio mas tiempo en ser atendido de lo que tarda en llegar el siguiente.<br>"
        f"<b>3. Factor de utilizacion:</b> El cociente rho = {datos_base['rho']:.4f} es estrictamente mayor que 1 "
        f"(rho &gt; 1). En la teoria de colas para sistemas de un solo servidor, rho &gt; 1 indica que "
        f"el sistema es <b>no estacionario y saturado</b>. La cola no puede estabilizarse y crecera "
        f"progresivamente durante periodos sostenidos de demanda."
    )

    if datos_base["rho"] > 1:
        st.error(
            f"Condicion de saturacion confirmada: rho = {datos_base['rho']:.4f} > 1. "
            "La tasa de arribo supera la capacidad del surtidor unico habilitado."
        )


# ============================================================
# TAB 2: INSTRUCCION 2 — TRANSFORMADA INVERSA
# ============================================================

with tab2:

    section_title(ICO_SHUFFLE, "Instruccion 2 — Generacion de variables aleatorias por transformada inversa")

    instr_context(
        "Generar variables aleatorias. Construir una distribucion acumulada y aplicar la transformada "
        "inversa, utilizando numeros aleatorios uniformes entre 0 y 1."
    )

    st.divider()

    # Como se construye la acumulada
    section_title(ICO_BOOK, "Como se construye la distribucion acumulada?", color="#0284c7")

    st.write(
        "A partir de las frecuencias observadas, calculamos la probabilidad relativa "
        r"$p_i = f_i / N$. Luego acumulamos sucesivamente estas probabilidades: "
        r"$F(x_k) = \sum_{i=1}^{k} p_i$. "
        "Cada valor discreto queda asociado a un intervalo continuo semiabierto "
        r"$[F(x_{k-1}), F(x_k))$ dentro del rango $[0, 1)$."
    )

    # Tablas con intervalos de U
    col_u1, col_u2 = st.columns(2)

    with col_u1:
        st.markdown("**Intervalos de U — Tiempos entre llegadas**")
        df_inv_ll = tabla_llegadas.copy()
        df_inv_ll["intervalo"] = [
            f"{li:.2f} <= U < {ls:.2f}"
            for li, ls in zip(df_inv_ll["limite_inferior"], df_inv_ll["limite_superior"])
        ]
        df_inv_ll_display = df_inv_ll[[
            "tiempo", "frecuencia", "probabilidad", "acumulada", "intervalo"
        ]].rename(columns={
            "tiempo": "Tiempo (min)",
            "frecuencia": "Frecuencia",
            "probabilidad": "p_i",
            "acumulada": "F(x)",
            "intervalo": "Intervalo de U asignado",
        })
        st.dataframe(df_inv_ll_display, use_container_width=True, hide_index=True)

    with col_u2:
        st.markdown("**Intervalos de U — Tiempos de servicio**")
        df_inv_sv = tabla_servicio.copy()
        df_inv_sv["intervalo"] = [
            f"{li:.2f} <= U < {ls:.2f}"
            for li, ls in zip(df_inv_sv["limite_inferior"], df_inv_sv["limite_superior"])
        ]
        df_inv_sv_display = df_inv_sv[[
            "tiempo", "frecuencia", "probabilidad", "acumulada", "intervalo"
        ]].rename(columns={
            "tiempo": "Tiempo (min)",
            "frecuencia": "Frecuencia",
            "probabilidad": "p_i",
            "acumulada": "F(x)",
            "intervalo": "Intervalo de U asignado",
        })
        st.dataframe(df_inv_sv_display, use_container_width=True, hide_index=True)

    st.divider()

    # Formulación matemática por tramos
    section_title(ICO_CALC, "Formulacion matematica de la transformada inversa", color="#0284c7")

    col_form_a, col_form_s = st.columns(2)

    with col_form_a:
        st.markdown("**Para los tiempos entre llegadas (A):**")
        st.latex(
            r"A = \begin{cases} "
            r"1 & 0.00 \le U < 0.20 \\ "
            r"2 & 0.20 \le U < 0.50 \\ "
            r"3 & 0.50 \le U < 0.80 \\ "
            r"4 & 0.80 \le U < 0.94 \\ "
            r"5 & 0.94 \le U < 1.00 "
            r"\end{cases}"
        )

    with col_form_s:
        st.markdown("**Para los tiempos de servicio (S):**")
        st.latex(
            r"S = \begin{cases} "
            r"2 & 0.00 \le U < 0.10 \\ "
            r"3 & 0.10 \le U < 0.30 \\ "
            r"4 & 0.30 \le U < 0.70 \\ "
            r"5 & 0.70 \le U < 0.90 \\ "
            r"6 & 0.90 \le U < 1.00 "
            r"\end{cases}"
        )

    st.divider()

    # Prueba interactiva con Slider
    section_title(ICO_SHUFFLE, "Simulador interactivo del metodo de la transformada inversa", color="#0284c7")

    st.write("Mueva el control deslizante para asignar un valor de U y verificar el mapeo resultante:")
    u_prueba = st.slider("Numero aleatorio uniforme U", 0.0000, 0.9999, 0.3500, 0.0001, format="%.4f")

    t_ll_res = transformar_uniforme(u_prueba, tabla_llegadas)
    t_sv_res = transformar_uniforme(u_prueba, tabla_servicio)

    col_res_u1, col_res_u2 = st.columns(2)
    col_res_u1.metric("Tiempo entre llegadas asignado", f"{t_ll_res} min")
    col_res_u2.metric("Tiempo de servicio asignado", f"{t_sv_res} min")

    st.divider()

    # Tabla de primeros 5 numeros aleatorios
    section_title(ICO_CHART, "Primeros cinco vehiculos: numeros aleatorios y transformaciones", color="#0284c7")

    primeros5_df = simulacion[[
        "vehiculo", "u_llegada", "entre_llegadas", "u_servicio", "servicio"
    ]].head(5).copy()

    primeros5_df.columns = [
        "Vehiculo",
        "U llegada",
        "Tiempo entre llegadas (min)",
        "U servicio",
        "Tiempo servicio (min)",
    ]

    st.dataframe(
        primeros5_df.style.format({
            "U llegada": "{:.5f}",
            "U servicio": "{:.5f}",
            "Tiempo entre llegadas (min)": "{:d}",
            "Tiempo servicio (min)": "{:d}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # Interpretacion de la tabla y lectura fila por fila
    section_title(ICO_BOOK, "Interpretacion de la tabla de transformacion", color="#0284c7")

    st.write(
        "Cada vehiculo utiliza dos numeros pseudoaleatorios uniformes entre 0 y 1: "
        "uno para determinar su tiempo entre llegadas y otro para determinar su tiempo de servicio.\n\n"
        "Los valores U no representan minutos directamente. Su funcion es seleccionar un intervalo "
        "dentro de la distribucion acumulada. El intervalo encontrado determina el tiempo generado."
    )

    st.markdown("**Como leer cada columna de la tabla:**")
    st.markdown(
        "- **Vehiculo:** Identifica al vehiculo simulado en secuencia cronologica.\n"
        "- **U llegada:** Numero uniforme generado $U \\sim \\text{Uniforme}(0, 1)$ para el tiempo de arribo.\n"
        "- **Tiempo entre llegadas (min):** Valor discreto obtenido tras aplicar la transformada inversa.\n"
        "- **U servicio:** Numero uniforme generado para el requerimiento de atencion.\n"
        "- **Tiempo de servicio (min):** Tiempo de despacho obtenido tras la transformada inversa."
    )

    # Detalle de lectura fila por fila de los 5 vehiculos generados
    st.markdown("**Lectura detallada de las primeras 5 filas simuladas:**")
    for _, fila_v in primeros5_df.iterrows():
        v_num = int(fila_v["Vehiculo"])
        u_ll = fila_v["U llegada"]
        t_ll = int(fila_v["Tiempo entre llegadas (min)"])
        u_sv = fila_v["U servicio"]
        t_sv = int(fila_v["Tiempo servicio (min)"])
        st.write(
            f"• Para el vehiculo {v_num}, el numero U = {u_ll:.5f} cayo en el tramo de llegada de {t_ll} minutos. "
            f"Por otra parte, el numero U = {u_sv:.5f} selecciono el intervalo de servicio de {t_sv} minutos."
        )

    st.info(
        "En resumen: U no representa directamente un tiempo. El número U indica qué intervalo "
        "de la distribución acumulada debe seleccionarse y ese intervalo determina el tiempo generado."
    )

    st.divider()

    # Ejemplo 1 y Ejemplo 2 paso a paso con los primeros valores
    v1_row = primeros5_df.iloc[0]
    u1_ll = v1_row["U llegada"]
    t1_ll = int(v1_row["Tiempo entre llegadas (min)"])
    u1_sv = v1_row["U servicio"]
    t1_sv = int(v1_row["Tiempo servicio (min)"])

    col_ej1, col_ej2 = st.columns(2)

    with col_ej1:
        st.markdown("**Ejemplo 1 — Transformacion de un tiempo entre llegadas**")
        st.write(f"Para el primer vehiculo se genero el numero uniforme U = {u1_ll:.5f}.")
        st.write("Buscamos en la distribucion acumulada de llegadas el intervalo correspondiente:")

        # Buscar limites
        fila_ll_match = tabla_llegadas[
            (tabla_llegadas["limite_inferior"] <= u1_ll) & (u1_ll < tabla_llegadas["limite_superior"])
        ].iloc[0]
        li_1, ls_1 = fila_ll_match["limite_inferior"], fila_ll_match["limite_superior"]

        st.latex(rf"{li_1:.2f} \le {u1_ll:.5f} < {ls_1:.2f}")
        st.write(f"Por lo tanto, el numero pertenece al intervalo de tiempo entre llegadas de {t1_ll} minutos.")
        st.latex(rf"A_1 = {t1_ll}\text{{ min}}")

    with col_ej2:
        st.markdown("**Ejemplo 2 — Transformacion de un tiempo de servicio**")
        st.write(f"Para el primer servicio se genero el numero uniforme U = {u1_sv:.5f}.")
        st.write("Buscamos en la distribucion acumulada de servicio el intervalo correspondiente:")

        fila_sv_match = tabla_servicio[
            (tabla_servicio["limite_inferior"] <= u1_sv) & (u1_sv < tabla_servicio["limite_superior"])
        ].iloc[0]
        li_s1, ls_s1 = fila_sv_match["limite_inferior"], fila_sv_match["limite_superior"]

        st.latex(rf"{li_s1:.2f} \le {u1_sv:.5f} < {ls_s1:.2f}")
        st.write(f"Por lo tanto, el numero pertenece al intervalo de tiempo de servicio de {t1_sv} minutos.")
        st.latex(rf"S_1 = {t1_sv}\text{{ min}}")


# ============================================================
# TAB 3: INSTRUCCION 3 — ANALISIS DE DISTRIBUCION (CHI-CUADRADO)
# ============================================================

with tab3:

    section_title(ICO_DIST, "Instruccion 3 — Analisis de distribucion y prueba de bondad de ajuste")

    instr_context(
        "Realizar el analisis de distribucion. Plantear una hipotesis de distribucion, "
        "calcular las frecuencias esperadas y aplicar chi-cuadrado cuando se cumplan sus condiciones. "
        "Si no se ajusta una distribucion teorica, pueden utilizar una distribucion empirica discreta para simular, justificandolo."
    )

    st.divider()

    # Nivel de significancia y regla de decision
    section_title(ICO_CHECK, "Nivel de significancia y criterio de decision", color="#0284c7")

    st.write(
        "Utilizaremos un nivel de significancia del 5 % (alpha = 0.05). "
        "Despues de calcular el p-valor correspondiente al estadistico chi-cuadrado, se toma la decision estadistica:"
    )

    st.latex(r"\alpha = 0.05")
    st.latex(r"p < \alpha \implies \text{Rechazar } H_0 \quad (\text{La distribucion teorica no ajusta})")
    st.latex(r"p \ge \alpha \implies \text{No rechazar } H_0 \quad (\text{No hay evidencia para descartar el modelo})")

    st.divider()

    # Hipótesis estadísticas lado a lado (H0 Izquierda vs H1 Derecha)
    section_title(ICO_CHECK, "Planteamiento de hipotesis estadisticas", color="#0284c7")

    col_h0, col_h1 = st.columns(2)

    with col_h0:
        st.markdown(
            '<div class="h0-block">'
            '<p style="font-weight:700;font-size:1rem;color:#15803d;margin-bottom:6px;">'
            'H0 — Hipotesis nula</p>'
            '<p style="margin:0;color:#166534;font-size:0.92rem;line-height:1.5;">'
            'Los datos observados provienen de una <b>distribucion teorica de Poisson</b> '
            'con parametro lambda estimado a partir de la media muestral observada. '
            'Las discrepancias observadas se deben exclusivamente a variaciones aleatorias del muestreo.'
            '</p></div>',
            unsafe_allow_html=True,
        )

    with col_h1:
        st.markdown(
            '<div class="h1-block">'
            '<p style="font-weight:700;font-size:1rem;color:#dc2626;margin-bottom:6px;">'
            'H1 — Hipotesis alternativa</p>'
            '<p style="margin:0;color:#991b1b;font-size:0.92rem;line-height:1.5;">'
            'Los datos observados <b>NO siguen una distribucion de Poisson</b>. '
            'La discrepancia entre las frecuencias observadas y las esperadas por el modelo teorico '
            'es estadisticamente significativa al nivel alpha = 0.05.'
            '</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Ejecutar pruebas chi-cuadrado de Poisson para llegadas y servicio
    res_poi_ll = ajuste_poisson(LLEGADAS)
    res_poi_sv = ajuste_poisson(SERVICIO)

    # --------------------------------------------------------
    # A. TIEMPOS ENTRE LLEGADAS
    # --------------------------------------------------------
    section_title(ICO_CHART, "A. Analisis de bondad de ajuste — Tiempos entre llegadas", color="#0284c7")

    st.markdown("**Parametro estimado de Poisson:**")
    st.write(
        "La distribucion Poisson necesita un parametro lambda. "
        "En este analisis se estima utilizando la media ponderada de los mismos datos observados (N = 50):"
    )
    st.latex(r"\hat{\lambda} = \frac{\sum x_i f_i}{N} = \frac{128}{50} = 2.5600\text{ min}")

    st.markdown("**Clases utilizadas en la prueba (Condicion de Cochran E_i >= 5):**")
    st.write(
        "Para construir la prueba se consideran las categorias discretas y la cola superior. "
        "Cuando alguna categoria tiene una frecuencia esperada menor que 5, se agrupa con una categoria contigua:"
    )
    st.latex(r"E_i \ge 5")
    st.write("Despues de realizar estas agrupaciones quedaron k = 5 clases finales: **0 + 1, 2, 3, 4, 5 + >=6**.")

    # Tabla observadas vs esperadas
    st.markdown("**Frecuencias observadas y esperadas:**")
    t_ll_display = res_poi_ll["tabla"].copy()
    t_ll_display.columns = ["Frec. observada (O_i)", "Frec. esperada (E_i)", "Clase combinada"]
    t_ll_display = t_ll_display[["Clase combinada", "Frec. observada (O_i)", "Frec. esperada (E_i)"]]
    t_ll_display["(O_i - E_i)^2 / E_i"] = (
        (t_ll_display["Frec. observada (O_i)"] - t_ll_display["Frec. esperada (E_i)"]) ** 2
        / t_ll_display["Frec. esperada (E_i)"]
    ).round(4)
    st.dataframe(t_ll_display, use_container_width=True, hide_index=True)
    st.caption("Σ O_i = 50; Σ E_i = 50.0000; minimo E_i despues de agrupar = 5.8491 >= 5.")

    # Ejemplo paso a paso clase 2
    with st.expander("Ver ejemplo paso a paso del calculo con la clase 2 (Llegadas)"):
        st.write("Tomemos como ejemplo la clase 2:")
        st.write("• En nuestros datos observamos O_i = 15 casos.")
        st.write("• Probabilidad teórica Poisson:")
        st.latex(r"P(X = 2) = \frac{e^{-2.5600} \cdot 2.5600^2}{2!} \approx 0.253312")
        st.write("• Frecuencia esperada:")
        st.latex(r"E_2 = N \cdot p_2 = 50 \cdot (0.253312) = 12.6656")
        st.write("• Contribucion de esta clase al chi-cuadrado:")
        st.latex(r"\frac{(O_2 - E_2)^2}{E_2} = \frac{(15 - 12.6656)^2}{12.6656} = 0.430250")

    # Grados de libertad y estadístico
    col_res_ll1, col_res_ll2, col_res_ll3, col_res_ll4 = st.columns(4)
    col_res_ll1.metric("chi^2 calculado", f"{res_poi_ll['chi2']:.4f}")
    col_res_ll2.metric("Clases finales (k)", "5")
    col_res_ll2.metric("Parametros estimados (q)", "1 (lambda)")
    col_res_ll3.metric("Grados de libertad (gl)", f"{res_poi_ll['gl']}")
    col_res_ll4.metric("p-valor", f"{res_poi_ll['pvalor']:.6f}")

    st.latex(r"gl = k - 1 - q = 5 - 1 - 1 = 3")
    st.latex(rf"\chi^2 = 1.0276 + 0.4303 + 1.6259 + 0.0010 + 1.3878 = {res_poi_ll['chi2']:.6f}")
    st.latex(rf"p = P(\chi_3^2 \ge {res_poi_ll['chi2']:.4f}) = {res_poi_ll['pvalor']:.6f} \quad (\approx 21.48\%)")

    st.success(
        f"Decision para tiempos entre llegadas: p = {res_poi_ll['pvalor']:.6f} >= alpha (0.05) -> NO RECHAZAR H0. "
        "No existe evidencia estadistica suficiente para descartar que las llegadas sigan una distribucion de Poisson. "
        "(Nota metodologica: no rechazar H0 no demuestra que sea Poisson; simplemente es compatible)."
    )

    st.divider()

    # --------------------------------------------------------
    # B. TIEMPOS DE SERVICIO
    # --------------------------------------------------------
    section_title(ICO_CHART, "B. Analisis de bondad de ajuste — Tiempos de servicio", color="#0284c7")

    st.markdown("**Parametro estimado de Poisson:**")
    st.latex(r"\hat{\lambda} = \frac{\sum x_i f_i}{N} = \frac{200}{50} = 4.0000\text{ min}")

    st.markdown("**Clases utilizadas en la prueba (Condicion de Cochran E_i >= 5):**")
    st.write(
        "Inicialmente se consideran 8 categorias. Despues de fusionar clases con frecuencia esperada "
        "menor que 5, quedan k = 6 clases finales: **0 + 1 + 2, 3, 4, 5, 6, >=7**."
    )

    # Tabla observadas vs esperadas de servicio
    t_sv_display = res_poi_sv["tabla"].copy()
    t_sv_display.columns = ["Frec. observada (O_i)", "Frec. esperada (E_i)", "Clase combinada"]
    t_sv_display = t_sv_display[["Clase combinada", "Frec. observada (O_i)", "Frec. esperada (E_i)"]]
    t_sv_display["(O_i - E_i)^2 / E_i"] = (
        (t_sv_display["Frec. observada (O_i)"] - t_sv_display["Frec. esperada (E_i)"]) ** 2
        / t_sv_display["Frec. esperada (E_i)"]
    ).round(4)
    st.dataframe(t_sv_display, use_container_width=True, hide_index=True)
    st.caption("Σ O_i = 50; Σ E_i = 50.0000; minimo E_i despues de agrupar = 5.2098 >= 5.")

    with st.expander("Ver ejemplo paso a paso del calculo con la clase 3 (Servicio)"):
        st.write("Tomemos como ejemplo la clase 3:")
        st.write("• En nuestros datos observamos O_i = 10 casos.")
        st.latex(r"P(X = 3) = \frac{e^{-4.0000} \cdot 4.0000^3}{3!} \approx 0.195367")
        st.latex(r"E_3 = 50 \cdot (0.195367) = 9.7683")
        st.latex(r"\frac{(O_3 - E_3)^2}{E_3} = \frac{(10 - 9.7683)^2}{9.7683} = 0.005494")

    col_res_sv1, col_res_sv2, col_res_sv3, col_res_sv4 = st.columns(4)
    col_res_sv1.metric("chi^2 calculado", f"{res_poi_sv['chi2']:.4f}")
    col_res_sv2.metric("Clases finales (k)", "6")
    col_res_sv2.metric("Parametros estimados (q)", "1 (lambda)")
    col_res_sv3.metric("Grados de libertad (gl)", f"{res_poi_sv['gl']}")
    col_res_sv4.metric("p-valor", f"{res_poi_sv['pvalor']:.6f}")

    st.latex(r"gl = k - 1 - q = 6 - 1 - 1 = 4")
    st.latex(rf"\chi^2 = 4.0051 + 0.0055 + 10.7170 + 0.6111 + 0.0084 + 5.5337 = {res_poi_sv['chi2']:.6f}")
    st.latex(rf"p = P(\chi_4^2 \ge {res_poi_sv['chi2']:.4f}) = {res_poi_sv['pvalor']:.6f} \quad (\approx 0.033\%)")

    st.error(
        f"Decision para tiempos de servicio: p = {res_poi_sv['pvalor']:.6f} < alpha (0.05) -> SE RECHAZA H0. "
        "Existe evidencia estadistica suficiente para afirmar que los tiempos de servicio NO siguen una distribucion de Poisson."
    )

    st.divider()

    # Decision final para la simulacion
    section_title(ICO_CHECK, "Decision metodologica para la simulacion", color="#0284c7")

    st.latex(r"P_{emp}(X = x_i) = \frac{f_i}{N}")

    interpretacion(
        "<b>Justificacion de uso de la distribucion empirica discreta:</b><br>"
        "La simulacion continuara utilizando la <b>distribucion empirica discreta</b> derivada directamente "
        "de las frecuencias observadas en campo, tanto para tiempos entre llegadas como para tiempos de servicio, "
        "mediante la transformada inversa de las Instrucciones 1 y 2.<br>"
        "Esta decision se fundamenta en que:<br>"
        "1. Para el tiempo de servicio se rechazo categóricamente el ajuste Poisson (p = 0.000334).<br>"
        "2. Para los tiempos entre llegadas, aunque no se rechazo Poisson (p = 0.2148), "
        "la distribucion empirica conserva fielmente la estructura y soporte discreto exacto de los datos muestreados "
        "sin introducir supuestos teoricos artificiales en colas infinitas.<br>"
        "El analisis de Poisson cumple la funcion academica de contrastar la hipotesis de ajuste, "
        "pero no altera el mecanismo generador empirico del modelo."
    )


# ============================================================
# TAB 4: INSTRUCCION 4 — CONSTRUCCION DEL MODELO
# ============================================================

with tab4:

    section_title(ICO_SIM, "Instruccion 4 — Construccion del modelo de simulacion")

    instr_context(
        "Construir el modelo. Programar o utilizar una entidad cliente (vehiculo), "
        "una cola FIFO y un servidor (surtidor). Inicializacion: servidor libre, cola vacia."
    )

    st.divider()

    # Metricas globales de la simulacion base
    section_title(ICO_CHART, "Indicadores principales de desempeno (Replica base)", color="#0284c7")

    col_sim_m1, col_sim_m2, col_sim_m3, col_sim_m4 = st.columns(4)
    col_sim_m1.metric("Espera media en cola", f"{resumen['espera_media']:.2f} min")
    col_sim_m2.metric("Espera maxima", f"{resumen['espera_maxima']:.2f} min")
    col_sim_m3.metric("Cola maxima observada", f"{resumen['cola_maxima']} vehiculos")
    col_sim_m4.metric("Utilizacion del surtidor", f"{resumen['utilizacion_observada'] * 100:.1f} %")

    st.divider()

    # Timeline Plotly
    section_title(ICO_SIM, "Cronograma y linea temporal de vehiculos (Gantt)", color="#0284c7")
    st.plotly_chart(
        grafico_timeline(simulacion, min(30, int(n_clientes))),
        use_container_width=True,
    )

    st.divider()

    # Reloj de simulacion interactivo
    section_title(ICO_SIM, "Reloj de simulacion e inspeccion de estado", color="#0284c7")

    t_reloj = st.slider(
        "Tiempo de simulacion (minutos)",
        min_value=0,
        max_value=int(math.ceil(resumen["horizonte"])),
        value=0,
        step=1,
    )

    estado_t = estado_en_tiempo(simulacion, t_reloj)

    c_st1, c_st2, c_st3 = st.columns(3)
    c_st1.metric("Vehiculos esperando en cola", len(estado_t["esperando"]))
    servidor_texto = f"V{estado_t['atendiendo'][0]}" if estado_t["atendiendo"] else "LIBRE"
    c_st2.metric("Estado del surtidor", servidor_texto)
    c_st3.metric("Vehiculos ya atendidos", len(estado_t["finalizados"]))

    # Cola FIFO
    st.markdown("**Estado actual de la cola FIFO:**")
    if estado_t["esperando"]:
        cola_str = " -> ".join([f"V{v}" for v in estado_t["esperando"]])
        st.code(f"COLA (FIFO): [Entrada] -> {cola_str} -> [Surtidor]")
    else:
        st.success("Cola vacia en este instante de tiempo.")

    st.divider()

    # Inspeccion de formulas de recurrencia por vehiculo
    section_title(ICO_CALC, "Inspeccion detallada por vehiculo y logica de simulacion", color="#0284c7")

    st.write(
        "La simulacion por eventos discretos actualiza el estado mediante las formulas de Lindley-Kiefer:"
    )
    st.latex(r"B_i = \max(T_i,\, D_{i-1}) \quad (\text{Hora de inicio de atencion})")
    st.latex(r"D_i = B_i + S_i \quad (\text{Hora de finalizacion y salida})")
    st.latex(r"W_{q,i} = B_i - T_i \quad (\text{Tiempo de espera en cola})")

    veh_sel = st.number_input(
        "Seleccione numero de vehiculo para inspeccionar",
        min_value=1,
        max_value=int(n_clientes),
        value=1,
        step=1,
    )

    fila_v_sel = simulacion.iloc[int(veh_sel) - 1]

    col_insp1, col_insp2 = st.columns(2)

    with col_insp1:
        st.markdown(f"**Tiempos cronologicos del Vehiculo {int(veh_sel)}:**")
        df_insp_ev = pd.DataFrame({
            "Evento": ["Llegada (T_i)", "Inicio de servicio (B_i)", "Salida del sistema (D_i)"],
            "Instante cronologico (min)": [
                int(fila_v_sel["llegada"]),
                int(fila_v_sel["inicio_servicio"]),
                int(fila_v_sel["salida"]),
            ]
        })
        st.dataframe(df_insp_ev, use_container_width=True, hide_index=True)

    with col_insp2:
        st.markdown(f"**Indicadores de servicio del Vehiculo {int(veh_sel)}:**")
        df_insp_ind = pd.DataFrame({
            "Variable": [
                "Tiempo entre llegadas",
                "Tiempo de servicio (S_i)",
                "Espera en cola (W_q,i)",
                "Tiempo total en sistema (W_i)",
                "Vehiculos en cola al llegar",
            ],
            "Duracion / Cantidad": [
                f"{int(fila_v_sel['entre_llegadas'])} min",
                f"{int(fila_v_sel['servicio'])} min",
                f"{int(fila_v_sel['espera_cola'])} min",
                f"{int(fila_v_sel['tiempo_sistema'])} min",
                f"{int(fila_v_sel['cola_al_llegar'])} vehiculos",
            ]
        })
        st.dataframe(df_insp_ind, use_container_width=True, hide_index=True)

    st.divider()

    # Registro de vehículos simulados (Tabla completa)
    section_title(ICO_CHART, "Registro de vehiculos simulados", color="#0284c7")

    with st.expander("Desplegar tabla completa de simulacion (Registro de eventos)"):
        st.dataframe(simulacion, use_container_width=True)


# ============================================================
# TAB 5: INSTRUCCION 5 — REPLICAS INDEPENDIENTES
# ============================================================

with tab5:

    section_title(ICO_REP, "Instruccion 5 — Ejecucion de replicas independientes")

    instr_context(
        "Ejecutar las replicas. Registrar los resultados de cada ejecucion con semillas aleatorias "
        "diferentes, cuando corresponda."
    )

    st.divider()

    section_title(ICO_CHART, "Resultados obtenidos en las replicas ejecutadas", color="#0284c7")

    st.dataframe(
        resultados_rep.style.format({
            "espera_media": "{:.2f} min",
            "espera_maxima": "{:.2f} min",
            "cola_maxima": "{:d}",
            "tiempo_medio_sistema": "{:.2f} min",
            "utilizacion_observada": "{:.1%}",
            "horizonte": "{:.2f} min",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    section_title(ICO_REP, "Grafico comparativo entre replicas", color="#0284c7")

    st.plotly_chart(
        grafico_replicas(resultados_rep),
        use_container_width=True,
    )

    st.divider()

    # Resumen preliminar del intervalo de confianza en réplicas
    ic_rep_pre = intervalo_confianza(resultados_rep["espera_media"], confianza)

    section_title(ICO_CHECK, "Resumen preliminar de convergencia", color="#0284c7")

    col_rep_p1, col_rep_p2, col_rep_p3 = st.columns(3)
    col_rep_p1.metric("Media global de espera", f"{ic_rep_pre['media']:.2f} min")
    col_rep_p2.metric(
        f"Intervalo t ({confianza*100:.0f} %)",
        f"[{ic_rep_pre['li']:.2f}, {ic_rep_pre['ls']:.2f}] min"
    )
    col_rep_p3.metric("Error relativo actual", f"{ic_rep_pre['error_relativo'] * 100:.2f} %")


# ============================================================
# TAB 6: INSTRUCCION 6 — ANALISIS ESTADISTICO COMPLETO
# ============================================================

with tab6:

    section_title(ICO_BOOK, "Instruccion 6 — Analisis estadistico de los resultados")

    instr_context(
        "Analizar estadisticamente los resultados. Calcular la media de espera, su intervalo de confianza, "
        "el error relativo y el numero de replicas requerido segun la precision solicitada."
    )

    st.divider()

    # Resultados obtenidos en las réplicas
    section_title(ICO_CHART, "Resultados obtenidos en las replicas", color="#0284c7")

    st.write(
        f"En la Instruccion 5 ejecutamos R = {int(n_replicas)} replicas independientes. "
        "Cada replica produjo un tiempo medio de espera en cola:"
    )
    st.latex(r"Y_r = \text{Tiempo medio de espera en cola de la replica } r")
    st.write(
        "Por lo tanto, nuestros datos estadisticos ya no son los vehiculos individuales. "
        "Ahora trabajamos con las medias obtenidas en las replicas independientes:"
    )

    medias_lista = resultados_rep["espera_media"].tolist()
    y_str = ", ".join([f"Y_{i+1} = {m:.2f}\\text{{ min}}" for i, m in enumerate(medias_lista)])
    st.latex(y_str)

    st.divider()

    # 1. Media de espera
    section_title(ICO_CALC, "1. Media de espera global", color="#0284c7")

    R_actual = len(medias_lista)
    suma_y = sum(medias_lista)
    media_global = suma_y / R_actual

    st.latex(r"\bar{Y} = \frac{1}{R} \sum_{r=1}^{R} Y_r")

    suma_str = " + ".join([f"{m:.2f}" for m in medias_lista])
    st.latex(rf"\bar{{Y}} = \frac{{{suma_str}}}{{{R_actual}}} = \frac{{{suma_y:.2f}}}{{{R_actual}}} = {media_global:.2f}\text{{ min}}")

    st.metric("Media global de espera en cola", f"{media_global:.2f} min")
    st.write(f"En promedio, considerando las {R_actual} replicas, un vehiculo espero aproximadamente {media_global:.2f} minutos en cola.")

    st.divider()

    # 2. Variabilidad entre replicas (Desviación estándar s_Y)
    section_title(ICO_CALC, "2. Variabilidad entre las replicas", color="#0284c7")

    s_y = float(np.std(medias_lista, ddof=1)) if R_actual > 1 else 0.0

    st.latex(r"s_Y = \sqrt{\frac{1}{R - 1} \sum_{r=1}^{R} (Y_r - \bar{Y})^2}")
    st.metric("Desviacion estandar entre replicas (s_Y)", f"{s_y:.4f} min")
    st.write(
        "Una desviacion mayor indica que las distintas ejecuciones pseudoaleatorias del modelo "
        "producen resultados mas dispersos debido a la naturaleza estocastica del sistema."
    )

    st.divider()

    # 3. Error estándar de la media
    section_title(ICO_CALC, "3. Error estandar de la media (SE)", color="#0284c7")

    se = s_y / math.sqrt(R_actual) if R_actual > 0 else 0.0

    st.latex(r"SE = \frac{s_Y}{\sqrt{R}}")
    st.latex(rf"SE = \frac{{{s_y:.4f}}}{{\sqrt{{{R_actual}}}}} = {se:.4f}\text{{ min}}")
    st.metric("Error estandar de la media (SE)", f"{se:.4f} min")

    st.divider()

    # 4. Valor crítico t de Student
    section_title(ICO_CALC, "4. Valor critico t de Student", color="#0284c7")

    gl_t = R_actual - 1
    alpha_val = 1 - confianza
    t_crit = float(t.ppf(1 - alpha_val / 2, df=gl_t)) if gl_t > 0 else np.nan

    st.write(
        f"Como tenemos una muestra pequena de replicas (R = {R_actual}) y la varianza poblacional "
        "es desconocida, utilizamos la distribucion t de Student:"
    )
    st.latex(r"gl = R - 1")
    st.latex(rf"gl = {R_actual} - 1 = {gl_t}")
    st.latex(rf"t_{{1 - \alpha/2,\, gl}} = t_{{{1 - alpha_val/2:.3f},\, {gl_t}}} = {t_crit:.4f}")
    st.metric(f"t critico (confianza {confianza*100:.0f} %)", f"{t_crit:.4f}")

    st.divider()

    # 5. Semiamplitud del intervalo (h)
    section_title(ICO_CALC, "5. Semiamplitud del intervalo (h)", color="#0284c7")

    h = t_crit * se

    st.latex(r"h = t_{\alpha/2,\, R-1} \cdot \frac{s_Y}{\sqrt{R}} = t_{\alpha/2,\, R-1} \cdot SE")
    st.latex(rf"h = ({t_crit:.4f}) \cdot ({se:.4f}) = {h:.2f}\text{{ min}}")
    st.metric("Semiamplitud (h)", f"{h:.2f} min")

    st.divider()

    # 6. Intervalo de confianza del nivel seleccionado
    section_title(ICO_CALC, f"6. Intervalo de confianza del {confianza*100:.0f} %", color="#0284c7")

    lim_inf = media_global - h
    lim_sup = media_global + h

    st.latex(r"IC = \bar{Y} \pm h")
    st.latex(rf"IC = {media_global:.2f} \pm {h:.2f}")
    st.latex(rf"IC = [{lim_inf:.2f},\, {lim_sup:.2f}]\text{{ min}}")

    c_ic1, c_ic2, c_ic3 = st.columns(3)
    c_ic1.metric("Limite inferior", f"{lim_inf:.2f} min")
    c_ic2.metric("Media global", f"{media_global:.2f} min")
    c_ic3.metric("Limite superior", f"{lim_sup:.2f} min")

    st.write(
        "Este intervalo representa la incertidumbre estadistica asociada a nuestra estimacion "
        "del tiempo medio de espera en cola del sistema."
    )

    st.divider()

    # 7. Error relativo
    section_title(ICO_CALC, "7. Error relativo obtenido vs deseado", color="#0284c7")

    e_rel = h / abs(media_global) if media_global != 0 else np.nan

    st.latex(r"e_r = \frac{h}{|\bar{Y}|}")
    st.latex(rf"e_r = \frac{{{h:.4f}}}{{{abs(media_global):.4f}}} = {e_rel:.4f} \quad ({e_rel * 100:.2f}\%)")

    c_er1, c_er2 = st.columns(2)
    c_er1.metric("Error relativo obtenido", f"{e_rel * 100:.2f} %")
    c_er2.metric("Error relativo maximo permitido", f"{precision * 100:.2f} %")

    if e_rel <= precision:
        st.success(
            f"El error relativo obtenido ({e_rel * 100:.2f} %) CUMPLE con la precision deseada (<= {precision * 100:.0f} %)."
        )
    else:
        st.warning(
            f"Las {R_actual} replicas iniciales TODAVIA NO permiten alcanzar el error relativo maximo del {precision * 100:.0f} %. "
            f"Error actual: {e_rel * 100:.2f} %. Se debe estimar cuantas replicas serian necesarias."
        )

    st.divider()

    # 8. Número de réplicas requerido
    section_title(ICO_CALC, "8. Numero de replicas requerido (Precision)", color="#0284c7")

    R_necesarias = replicas_requeridas(media_global, s_y, precision, confianza)
    R_adicionales = max(0, R_necesarias - R_actual)

    st.latex(r"R^* \approx \left( \frac{t \cdot s_Y}{e_{max} \cdot |\bar{Y}|} \right)^2")
    st.write("Dado que el valor critico t depende de los grados de libertad, el calculo se realiza iterativamente:")

    c_rn1, c_rn2 = st.columns(2)
    c_rn1.metric("Replicas totales estimadas (R*)", f"{R_necesarias}")
    c_rn2.metric("Replicas adicionales estimadas", f"{R_adicionales}")

    st.write(
        f"Con la variabilidad observada en las {R_actual} replicas iniciales, se estiman aproximadamente "
        f"<b>{R_necesarias} replicas totales</b> para alcanzar un error relativo maximo del {precision * 100:.0f} %. "
        f"Esto implica ejecutar unas {R_adicionales} replicas adicionales."
    )

    st.divider()

    # Resumen y resultado formal de la Instrucción 6
    section_title(ICO_CHECK, "Resultado de la Instruccion 6", color="#0284c7")

    interpretacion(
        f"<b>Resumen formal:</b><br>"
        f"• Las {R_actual} replicas iniciales produjeron un tiempo medio de espera global de <b>{media_global:.2f} minutos</b>.<br>"
        f"• Con un nivel de confianza del {confianza * 100:.0f} %, el intervalo estimado es de <b>[{lim_inf:.2f}, {lim_sup:.2f}] minutos</b>.<br>"
        f"• El error relativo obtenido es <b>{e_rel * 100:.2f} %</b> "
        f"({'adecuado' if e_rel <= precision else 'superior al maximo solicitado del ' + str(int(precision * 100)) + ' %'}).<br>"
        f"• Por consiguiente, se requieren <b>{R_necesarias} replicas totales</b> para garantizar la precision solicitada."
    )

    st.divider()

    # Interpretación de demanda vs capacidad y cola creciente
    section_title(ICO_BOOK, "Interpretacion operativa del sistema", color="#0284c7")

    col_interp_op1, col_interp_op2 = st.columns(2)

    with col_interp_op1:
        st.markdown("**La demanda supera la capacidad?**")
        st.write(
            f"El tiempo medio entre llegadas es de {datos_base['media_entre_llegadas']:.2f} min, "
            f"mientras que el servicio medio es de {datos_base['media_servicio']:.2f} min."
        )
        st.latex(rf"\rho = {datos_base['rho']:.4f} > 1")
        st.error("La demanda promedio supera la capacidad del unico surtidor habilitado.")

    with col_interp_op2:
        st.markdown("**Una cola creciente significa que el programa esta mal?**")
        st.write(
            "No necesariamente. En este escenario los vehiculos arriban a una tasa mayor de la que "
            "el servidor puede despachar. Por tanto, una cola creciente es plenamente compatible "
            "con los datos observados y las leyes de la teoria de colas."
        )

    st.divider()

    # Escenarios futuros
    section_title(ICO_BOOK, "Escenarios futuros propuestos", color="#0284c7")
    st.write(
        "En la siguiente etapa se recomienda evaluar mediante la misma metodologia de simulacion:\n"
        "1. **Surtidor actual:** Linea base con sobrecarga evidente (rho > 1).\n"
        "2. **Servicio mas rapido:** Disminuir el tiempo medio de atencion para lograr rho < 1.\n"
        "3. **Dos surtidores:** Sistema de dos servidores en paralelo para duplicar la tasa mu.\n"
        "4. **Incremento de capacidad en hora pico:** Disenar turnos de refuerzo selectivos."
    )


# ============================================================
# TAB 7: PREGUNTAS TECNICAS DE VERIFICACION
# ============================================================

with tab7:

    section_title(ICO_TECH, "Preguntas tecnicas de verificacion")

    instr_context(
        "Verificacion paso a paso de los requerimientos tecnicos y metodologicos del ejercicio. "
        "En la Pregunta 2 puede interactuar cambiando o ingresando valores manualmente."
    )

    st.divider()

    # --------------------------------------------------------
    # PREGUNTA 1
    # --------------------------------------------------------
    st.markdown(
        '<div class="q-block">'
        '<div class="q-label">Pregunta Tecnica 1</div>'
        '<div class="q-text">Presenten los primeros cinco numeros aleatorios utilizados '
        'y muestren como transformaron al menos dos de ellos en tiempos entre llegadas y tiempos de servicio.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    p1_tabla = simulacion[[
        "vehiculo", "u_llegada", "entre_llegadas", "u_servicio", "servicio"
    ]].head(5).copy()
    p1_tabla.columns = [
        "Vehiculo", "U llegada", "Tiempo entre llegadas (min)",
        "U servicio", "Tiempo servicio (min)"
    ]

    st.dataframe(
        p1_tabla.style.format({
            "U llegada": "{:.5f}",
            "U servicio": "{:.5f}",
            "Tiempo entre llegadas (min)": "{:d}",
            "Tiempo servicio (min)": "{:d}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    st.write(
        "**Explicacion detallada de la transformacion para dos casos:**\n\n"
        f"1. **Tiempo entre llegadas del Vehiculo 1:** Se genero el numero uniforme $U = {p1_tabla.iloc[0]['U llegada']:.5f}$. "
        "Buscando en la distribucion acumulada de llegadas, este valor cae en el intervalo "
        f"$[0.00, 0.20)$, por lo que le corresponde exactamente **{p1_tabla.iloc[0]['Tiempo entre llegadas (min)']} minuto**.\n"
        f"2. **Tiempo de servicio del Vehiculo 1:** Se genero el numero uniforme $U = {p1_tabla.iloc[0]['U servicio']:.5f}$. "
        "Buscando en la distribucion acumulada de servicio, este valor cae en el intervalo "
        f"donde le corresponde un tiempo de servicio de **{p1_tabla.iloc[0]['Tiempo servicio (min)']} minutos**."
    )

    st.divider()

    # --------------------------------------------------------
    # PREGUNTA 2: RECONSTRUCCION DE 3 CLIENTES CONSECUTIVOS CON EDICION WEB
    # --------------------------------------------------------
    st.markdown(
        '<div class="q-block">'
        '<div class="q-label">Pregunta Tecnica 2 (Interactivo: Seleccionar o Modificar Valores)</div>'
        '<div class="q-text">Seleccionen tres clientes consecutivos y reconstruyan manualmente '
        'su hora de llegada, inicio de servicio, finalizacion y tiempo de espera.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "Puede seleccionar que vehiculo inicial observar de la simulacion actual, "
        "o activar la casilla de edicion para ingresar sus propios valores personalizados y ver el calculo paso a paso:"
    )

    col_p2_cfg1, col_p2_cfg2 = st.columns(2)

    with col_p2_cfg1:
        veh_p2_ini = st.number_input(
            "Vehiculo inicial de la simulacion",
            min_value=1,
            max_value=max(1, int(n_clientes) - 2),
            value=1,
            step=1,
            key="p2_veh_ini",
        )

    with col_p2_cfg2:
        usar_personalizados = st.checkbox(
            "Modificar/personalizar valores manualmente desde la web",
            value=False,
            key="p2_custom_check",
        )

    # Extraer valores base de la simulación
    sub_sim = simulacion.iloc[veh_p2_ini - 1 : veh_p2_ini + 2].copy().reset_index(drop=True)

    # Hora de salida del cliente anterior al primero del trío
    if veh_p2_ini == 1:
        d_previo_default = 0
    else:
        d_previo_default = int(simulacion.iloc[veh_p2_ini - 2]["salida"])

    if usar_personalizados:
        st.markdown("**Panel de modificacion de valores de entrada para los 3 vehiculos:**")
        col_c0, col_c1, col_c2, col_c3 = st.columns(4)

        with col_c0:
            d_previo = st.number_input(
                f"Salida vehiculo anterior (D_{veh_p2_ini-1})",
                min_value=0,
                value=d_previo_default,
                step=1,
                key="p2_d_previo",
            )

        with col_c1:
            st.markdown(f"**Vehiculo {veh_p2_ini}**")
            t_ll_1 = st.number_input(f"Llegada T_{veh_p2_ini}", min_value=0, value=int(sub_sim.iloc[0]["llegada"]), step=1, key="p2_t1")
            s_1 = st.number_input(f"Servicio S_{veh_p2_ini}", min_value=1, value=int(sub_sim.iloc[0]["servicio"]), step=1, key="p2_s1")

        with col_c2:
            st.markdown(f"**Vehiculo {veh_p2_ini+1}**")
            t_ll_2 = st.number_input(f"Llegada T_{veh_p2_ini+1}", min_value=t_ll_1, value=int(sub_sim.iloc[1]["llegada"]), step=1, key="p2_t2")
            s_2 = st.number_input(f"Servicio S_{veh_p2_ini+1}", min_value=1, value=int(sub_sim.iloc[1]["servicio"]), step=1, key="p2_s2")

        with col_c3:
            st.markdown(f"**Vehiculo {veh_p2_ini+2}**")
            t_ll_3 = st.number_input(f"Llegada T_{veh_p2_ini+2}", min_value=t_ll_2, value=int(sub_sim.iloc[2]["llegada"]), step=1, key="p2_t3")
            s_3 = st.number_input(f"Servicio S_{veh_p2_ini+2}", min_value=1, value=int(sub_sim.iloc[2]["servicio"]), step=1, key="p2_s3")

        # Recalcular matematicamente
        b_1 = max(t_ll_1, d_previo)
        d_1 = b_1 + s_1
        w_1 = b_1 - t_ll_1

        b_2 = max(t_ll_2, d_1)
        d_2 = b_2 + s_2
        w_2 = b_2 - t_ll_2

        b_3 = max(t_ll_3, d_2)
        d_3 = b_3 + s_3
        w_3 = b_3 - t_ll_3

        df_recon = pd.DataFrame([
            {"Vehiculo": veh_p2_ini, "Llegada (T_i)": t_ll_1, "Servicio (S_i)": s_1, "Inicio servicio (B_i)": b_1, "Salida (D_i)": d_1, "Espera cola (W_q,i)": w_1},
            {"Vehiculo": veh_p2_ini + 1, "Llegada (T_i)": t_ll_2, "Servicio (S_i)": s_2, "Inicio servicio (B_i)": b_2, "Salida (D_i)": d_2, "Espera cola (W_q,i)": w_2},
            {"Vehiculo": veh_p2_ini + 2, "Llegada (T_i)": t_ll_3, "Servicio (S_i)": s_3, "Inicio servicio (B_i)": b_3, "Salida (D_i)": d_3, "Espera cola (W_q,i)": w_3},
        ])
    else:
        d_previo = d_previo_default
        df_recon = sub_sim[[
            "vehiculo", "llegada", "servicio", "inicio_servicio", "salida", "espera_cola"
        ]].copy()
        df_recon.columns = [
            "Vehiculo", "Llegada (T_i)", "Servicio (S_i)", "Inicio servicio (B_i)", "Salida (D_i)", "Espera cola (W_q,i)"
        ]

    st.markdown("**Tabla reconstruida paso a paso:**")
    st.dataframe(df_recon, use_container_width=True, hide_index=True)

    # Desarrollo matemático detallado
    st.markdown("**Desarrollo manual detallado de cada vehiculo:**")

    r1 = df_recon.iloc[0]
    r2 = df_recon.iloc[1]
    r3 = df_recon.iloc[2]

    st.write(
        f"**Vehiculo {int(r1['Vehiculo'])}:**\n"
        f"- Hora de llegada: $T = {int(r1['Llegada (T_i)'])} $\n"
        f"- Inicio de servicio: $B = \\max(T, D_{{previo}}) = \\max({int(r1['Llegada (T_i)'] )}, {int(d_previo)}) = {int(r1['Inicio servicio (B_i)'])} $\n"
        f"- Salida: $D = B + S = {int(r1['Inicio servicio (B_i)'])} + {int(r1['Servicio (S_i)'])} = {int(r1['Salida (D_i)'])} $\n"
        f"- Espera en cola: $W_q = B - T = {int(r1['Inicio servicio (B_i)'])} - {int(r1['Llegada (T_i)'])} = {int(r1['Espera cola (W_q,i)'])} $ min"
    )

    st.write(
        f"**Vehiculo {int(r2['Vehiculo'])}:**\n"
        f"- Hora de llegada: $T = {int(r2['Llegada (T_i)'])} $\n"
        f"- Inicio de servicio: $B = \\max(T, D_{{{int(r1['Vehiculo'])}}}) = \\max({int(r2['Llegada (T_i)'] )}, {int(r1['Salida (D_i)'])}) = {int(r2['Inicio servicio (B_i)'])} $\n"
        f"- Salida: $D = B + S = {int(r2['Inicio servicio (B_i)'])} + {int(r2['Servicio (S_i)'])} = {int(r2['Salida (D_i)'])} $\n"
        f"- Espera en cola: $W_q = B - T = {int(r2['Inicio servicio (B_i)'])} - {int(r2['Llegada (T_i)'])} = {int(r2['Espera cola (W_q,i)'])} $ min"
    )

    st.write(
        f"**Vehiculo {int(r3['Vehiculo'])}:**\n"
        f"- Hora de llegada: $T = {int(r3['Llegada (T_i)'])} $\n"
        f"- Inicio de servicio: $B = \\max(T, D_{{{int(r2['Vehiculo'])}}}) = \\max({int(r3['Llegada (T_i)'] )}, {int(r2['Salida (D_i)'])}) = {int(r3['Inicio servicio (B_i)'])} $\n"
        f"- Salida: $D = B + S = {int(r3['Inicio servicio (B_i)'])} + {int(r3['Servicio (S_i)'])} = {int(r3['Salida (D_i)'])} $\n"
        f"- Espera en cola: $W_q = B - T = {int(r3['Inicio servicio (B_i)'])} - {int(r3['Llegada (T_i)'])} = {int(r3['Espera cola (W_q,i)'])} $ min"
    )

    st.divider()

    # --------------------------------------------------------
    # PREGUNTA 3: EFECTO DE CAMBIAR LA SEMILLA
    # --------------------------------------------------------
    st.markdown(
        '<div class="q-block">'
        '<div class="q-label">Pregunta Tecnica 3</div>'
        '<div class="q-text">Que resultados cambian al utilizar otra semilla aleatoria '
        'y cuales deberian permanecer aproximadamente estables?</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    sem_alt_val = st.number_input(
        "Semilla alternativa para prueba comparativa",
        min_value=0,
        value=int(semilla) + 777,
        step=1,
        key="p3_sem_comp",
    )

    sim_alt_p3 = simular_gasolinera(int(n_clientes), int(sem_alt_val))
    res_alt_p3 = resumen_simulacion(sim_alt_p3)

    col_cmp1, col_cmp2 = st.columns(2)

    with col_cmp1:
        st.markdown(f"**Corrida base (Semilla {int(semilla)}):**")
        st.metric("Espera media", f"{resumen['espera_media']:.2f} min")
        st.metric("Espera maxima", f"{resumen['espera_maxima']:.2f} min")
        st.metric("Cola maxima", f"{resumen['cola_maxima']} vehiculos")
        st.metric("Utilizacion observada", f"{resumen['utilizacion_observada']*100:.1f} %")

    with col_cmp2:
        st.markdown(f"**Corrida alternativa (Semilla {int(sem_alt_val)}):**")
        st.metric(
            "Espera media",
            f"{res_alt_p3['espera_media']:.2f} min",
            delta=f"{res_alt_p3['espera_media'] - resumen['espera_media']:.2f}",
        )
        st.metric(
            "Espera maxima",
            f"{res_alt_p3['espera_maxima']:.2f} min",
            delta=f"{res_alt_p3['espera_maxima'] - resumen['espera_maxima']:.2f}",
        )
        st.metric(
            "Cola maxima",
            f"{res_alt_p3['cola_maxima']} vehiculos",
            delta=res_alt_p3["cola_maxima"] - resumen["cola_maxima"],
        )
        st.metric(
            "Utilizacion observada",
            f"{res_alt_p3['utilizacion_observada']*100:.1f} %",
            delta=f"{(res_alt_p3['utilizacion_observada'] - resumen['utilizacion_observada'])*100:.1f} %",
        )

    st.write(
        "**Explicacion detallada:**\n\n"
        "• **Resultados que cambian fuertemente:** Las metricas puntuales y extremas asociadas a una sola replica, "
        "tales como la secuencia exacta de vehiculos, la **espera maxima observada** y la **cola maxima instantanea**. "
        "Estos valores extremos son altamente sensibles a rachas locales de numeros aleatorios.\n\n"
        "• **Resultados que permanecen aproximadamente estables:** La **utilizacion del servidor** "
        "(que tiende a mantenerse cerca del 99 - 100 % por estar en saturacion rho > 1) y la "
        "**media global estimada a traves de multiples replicas independientes**. "
        "La teoria estocastica garantiza que al promediar varias corridas, las fluctuaciones aleatorias "
        "se compensan mutuamente convergiendo hacia la esperanza matematica del sistema."
    )

    st.divider()

    # --------------------------------------------------------
    # PREGUNTA 4: INTERVALO DE CONFIANZA
    # --------------------------------------------------------
    st.markdown(
        '<div class="q-block">'
        '<div class="q-label">Pregunta Tecnica 4</div>'
        '<div class="q-text">Expliquen como obtuvieron el intervalo de confianza '
        'y que significa para el tiempo medio de espera.</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.write(
        "**Procedimiento metodologico de obtencion:**\n"
        r"1. Se ejecutan $R$ réplicas independientes, cada una simulando $N$ vehículos con una semilla distinta." + "\n"
        r"2. Se registra la media de espera de cada réplica: $Y_1, Y_2, \dots, Y_R$." + "\n"
        r"3. Se calcula la media muestral global $\bar{Y}$ y la desviación estándar muestral entre réplicas $s_Y$ con $R - 1$ grados de libertad." + "\n"
        r"4. Se obtiene el valor crítico $t_{\alpha/2,\, R-1}$ de la distribución t de Student para el nivel de confianza elegido (95 %)." + "\n"
        r"5. Se calcula la semiamplitud del intervalo: $h = t_{\alpha/2,\, R-1} \cdot \frac{s_Y}{\sqrt{R}}$." + "\n"
        r"6. El intervalo queda definido por $[\bar{Y} - h,\, \bar{Y} + h]$."
    )

    st.latex(
        rf"IC_{{95\%}} = [{lim_inf:.2f},\, {lim_sup:.2f}]\text{{ min}}"
    )

    st.write(
        "**Significado estadistico e interpretacion:**\n\n"
        f"El intervalo significa que, bajo la distribucion empirica y condiciones del modelo, "
        f"tenemos un 95 % de confianza estadistica de que el verdadero tiempo medio de espera "
        f"en cola para 100 vehiculos se encuentra contenido entre **{lim_inf:.2f} y {lim_sup:.2f} minutos**.\n"
        "Desde la interpretacion frecuentista, si repitiesemos este experimento de 5 replicas 100 veces, "
        "aproximadamente 95 de los intervalos construidos contendrian a la media poblacional real."
    )

    st.divider()

    # --------------------------------------------------------
    # PREGUNTA 5: DUPLICAR EL NUMERO DE REPLICAS
    # --------------------------------------------------------
    st.markdown(
        '<div class="q-block">'
        '<div class="q-label">Pregunta Tecnica 5</div>'
        '<div class="q-text">Si duplicaran el numero de replicas, '
        'que esperan que suceda con la amplitud del intervalo de confianza?</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    R_doble = int(n_replicas) * 2
    res_doble_p5, _ = ejecutar_replicas(R_doble, int(n_clientes), int(semilla))
    ic_doble_p5 = intervalo_confianza(res_doble_p5["espera_media"], confianza)

    amp_orig = lim_sup - lim_inf
    amp_doble = ic_doble_p5["ls"] - ic_doble_p5["li"]
    reduccion_teorica = 1 - (1 / math.sqrt(2))

    col_d1, col_d2 = st.columns(2)
    col_d1.metric(f"Amplitud con R = {int(n_replicas)} replicas", f"{amp_orig:.2f} min")
    col_d2.metric(
        f"Amplitud con 2R = {R_doble} replicas",
        f"{amp_doble:.2f} min",
        delta=f"{amp_doble - amp_orig:.2f} min",
    )

    st.write(
        "**Fundamentacion matematica y teorica:**\n\n"
        r"La semiamplitud del intervalo de confianza depende inversamente de la raíz cuadrada del número de réplicas:"
    )
    st.latex(r"h \propto \frac{1}{\sqrt{R}}")
    st.write(
        r"Si se duplica el número de réplicas ($R \to 2R$), la nueva semiamplitud esperada es:"
    )
    st.latex(
        r"h_{nuevo} \approx \frac{h_{original}}{\sqrt{2}} \approx 0.7071 \cdot h_{original}"
    )
    st.write(
        f"Por lo tanto, se espera que la **amplitud del intervalo se reduzca aproximadamente un 29.3 %** "
        f"({reduccion_teorica*100:.1f} %). En nuestra prueba con 2R = {R_doble} réplicas, la amplitud "
        f"pasó de {amp_orig:.2f} min a {amp_doble:.2f} min, confirmando que mayor esfuerzo computacional "
        "aumenta la precisión estadística del estimador."
    )


# ============================================================
# TAB 8: PREGUNTAS DE PENSAMIENTO CRITICO
# ============================================================

with tab8:

    section_title(ICO_BRAIN, "Preguntas de pensamiento critico")

    instr_context(
        "Analisis reflexivo y justificacion de decisiones en simulacion. "
        "Respuestas contextualizadas en el Escenario 2 (Gasolinera, surtidor unico, FIFO)."
    )

    st.divider()

    rho_act = datos_base["rho"]
    w_med_act = resumen["espera_media"]
    w_max_act = resumen["espera_maxima"]

    # 1. Error de programación vs Demanda > Capacidad
    st.markdown(
        '<div class="crit-block">'
        '<div class="crit-q">Pensamiento Critico · Pregunta 1</div>'
        '<div class="crit-title">Si el modelo produce una cola creciente, '
        'como distinguirian entre un error de programacion y una demanda superior a la capacidad del servidor?</div>'
        '<div class="crit-a">'
        '<b>Respuesta fundamentada:</b><br>'
        'Para distinguir entre ambas situaciones se realiza un contraste riguroso entre la teoria analitica de colas '
        'y el codigo de simulacion:<br>'
        '1. <b>Calculo analitico de la tasa de utilizacion:</b> Se calculan las tasas medias observadas '
        f'lambda = {datos_base["lambda_min"]:.4f} veh/min y mu = {datos_base["mu_min"]:.4f} veh/min. '
        f'En nuestro caso, rho = lambda / mu = {rho_act:.4f} > 1. '
        'Cuando rho > 1, la ley de conservacion del flujo establece que el sistema no puede alcanzar un estado '
        'estacionario estable; por definicion, la cola DEBE crecer de forma acumulativa en el tiempo. '
        'Por ende, una cola creciente es el comportamiento matematicamente correcto y esperado para estos datos.<br>'
        '2. <b>Deteccion de errores de logica:</b> Si el calculo arrojara rho < 1 y aun asi la cola creciera '
        'indefinidamente, existiria un error de programacion, tal como fallos en el reloj de eventos, '
        'vehiculos que nunca liberan el servidor, condiciones de carrera en el bucle FIFO o calculos incorrectos '
        'del tiempo de inicio B_i = max(T_i, D_{i-1}).'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # 2. IC estrecho vs Validez de la realidad
    st.markdown(
        '<div class="crit-block">'
        '<div class="crit-q">Pensamiento Critico · Pregunta 2</div>'
        '<div class="crit-title">Puede un modelo tener un intervalo de confianza muy estrecho '
        'y, sin embargo, representar mal la realidad?</div>'
        '<div class="crit-a">'
        '<b>Respuesta fundamentada:</b><br>'
        '<b>Si, totalmente.</b> Es fundamental distinguir entre <i>precision estadistica</i> (varianza muestral baja) '
        'y <i>validez de modelo</i> (adecuacion a la realidad):<br>'
        '• Un intervalo de confianza estrecho unicamente certifica que el modelo computacional es <b>consistente</b> '
        'consigo mismo: al correr muchas replicas con las mismas premisas, los resultados varian muy poco entre si.<br>'
        '• Sin embargo, si los datos observados fueron recolectados con sesgo (por ejemplo, mediciones hechas '
        'solo en hora valle o durante una lluvia atipica), si los clientes reales desisten tras 10 minutos de cola '
        '(fenomeno de abandono no modelado), o si el servidor cambia su velocidad segun la fatiga, '
        'el modelo tendra alta precision matematica pero sera <b>invalido externamente</b>. '
        'En simulacion esto se resume como: <i>"es mejor tener una respuesta aproximada para el problema correcto '
        'que una respuesta exacta con intervalo estrecho para el problema equivocado"</i>.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # 3. Réplica favorable vs desfavorables
    st.markdown(
        '<div class="crit-block">'
        '<div class="crit-q">Pensamiento Critico · Pregunta 3</div>'
        '<div class="crit-title">Si una replica da un resultado favorable y otra desfavorable, '
        'cual presentarian a la gerencia? Por que?</div>'
        '<div class="crit-a">'
        '<b>Respuesta fundamentada:</b><br>'
        '<b>No se debe presentar ninguna de las dos replicas de forma individual.</b> '
        'A la gerencia se le debe presentar el <b>intervalo de confianza conjunto</b> '
        f'[{lim_inf:.2f}, {lim_sup:.2f}] min y la distribucion probabilistica de los escenarios evaluados.<br>'
        '• <b>Justificacion metodologica:</b> Presentar solo la replica favorable constituiria un sesgo '
        'de optimismo y una falta de etica tecnica, induciendo a la gerencia a subestimar el riesgo operativo '
        'y desestimar inversiones necesarias. Presentar solo la desfavorable incurriria en alarmismo innecesario.<br>'
        '• <b>Recomendacion gerencial:</b> La toma de decisiones corporativas exige conocer el valor esperado '
        f'({media_global:.2f} min), el rango de dispersion probable y el peor escenario con percentil 95 %, '
        'lo que permite cuantificar el costo del riesgo y diseñar planes de contingencia dimensionados.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # 4. Qué cambiar primero para reducir la espera
    st.markdown(
        '<div class="crit-block">'
        '<div class="crit-q">Pensamiento Critico · Pregunta 4</div>'
        '<div class="crit-title">Que cambiarian primero para reducir la espera: '
        'el tiempo de servicio, la politica de atencion o el numero de servidores? Que evidencia necesitarian?</div>'
        '<div class="crit-a">'
        '<b>Respuesta fundamentada:</b><br>'
        f'Dado que el sistema actual esta sobrecargado con rho = {rho_act:.4f} > 1, la prioridad numero uno es '
        'reducir la utilizacion por debajo de 1. El orden de evaluacion recomendado es:<br>'
        '1. <b>Primera opcion (Operativa / Bajo Costo): Reducir el tiempo de servicio.</b> '
        'Agilizar cobros (pago electronico anticipado, desvincular cobro de carga) para bajar el servicio medio '
        'de 4.00 min a 2.00 min. Con un servicio de 2.00 min, rho bajaria a 2.00 / 2.56 = 0.78 < 1, '
        'estabilizando la cola de raiz sin necesidad de construir infraestructura fisica.<br>'
        '2. <b>Segunda opcion (Inversion de Capital): Habilitar un segundo servidor (surtidor).</b> '
        'Si los tiempos de servicio no se pueden reducir por razones de seguridad o despacho de combustible, '
        'la unica forma de evitar colas infinitas es pasar a un esquema M/M/2, duplicando la tasa mu a 30 veh/h, '
        'lo que dejaria rho = 23.44 / 30 = 0.78.<br>'
        '3. <b>Politica de atencion:</b> Cambiar FIFO por otra disciplina (como SPT) no resuelve el problema '
        'estructural de fondo si rho > 1, solo redistribuiria inequitativamente la espera entre clientes.<br>'
        '• <b>Evidencia necesaria:</b> Un analisis de costo-beneficio comparando corridas de simulacion '
        'con diferentes reducciones de tiempo de servicio vs los costos de instalacion de una nueva bomba.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    # 5. Media disminuye pero máximo aumenta
    st.markdown(
        '<div class="crit-block">'
        '<div class="crit-q">Pensamiento Critico · Pregunta 5</div>'
        '<div class="crit-title">Si la media de espera disminuye, pero aumenta el tiempo maximo de espera, '
        'considerarian que la modificacion fue exitosa?</div>'
        '<div class="crit-a">'
        '<b>Respuesta fundamentada:</b><br>'
        '<b>No se puede catalogar como exitosa a priori; depende directamente de la politica de nivel de servicio (SLA) '
        'y del perfil de clientes del negocio:</b><br>'
        '• <b>Perspectiva global de eficiencia:</b> Si el objetivo es el rendimiento agregado del surtidor, '
        'una reduccion de la media indica que el sistema evacua mas vehiculos en menor tiempo promedio.<br>'
        '• <b>Perspectiva de satisfaccion y riesgo de abandono:</b> Sin embargo, un incremento en la espera maxima '
        'indica un <b>aumento severo en la varianza</b> del proceso. En servicios al cliente, las personas tienen '
        'tolerancia limitada: un cliente que espere un tiempo excesivo abandonara la fila, presentara reclamos '
        'o dañara la reputacion de la estacion.<br>'
        '• <b>Criterio de evaluacion integral:</b> Para declarar exitosa la intervencion se deben evaluar '
        'los percentiles altos (P90 y P95). Si la reduccion media beneficia al 90 % de los clientes y el aumento '
        'del maximo afecta a casos residuales controlables, puede ser aceptable; pero si viola un acuerdo '
        'de servicio critico, la variabilidad añadida es perjudicial y la modificacion debe rediseñarse.'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Preguntas del escenario 2
    section_title(ICO_BRAIN, "Preguntas del Escenario 2 (Gasolinera con una sola estacion)", color="#6d28d9")

    preguntas_esc_def = [
        (
            "1. Que evidencias indicarian que la demanda supera la capacidad de atencion?",
            f"La evidencia matematica primaria es el factor de carga <b>rho = {rho_act:.4f} > 1</b> "
            f"(tasa de arribo {datos_base['lambda_hora']:.2f} veh/h frente a una capacidad de solo {datos_base['mu_hora']:.2f} veh/h). "
            f"Operativamente, esto se evidencia en: (a) la cola maxima alcanza {resumen['cola_maxima']} vehiculos "
            f"y no se vacia, (b) la utilizacion del surtidor es del {resumen['utilizacion_observada']*100:.1f} % (saturacion casi total), "
            f"y (c) los tiempos de espera aumentan conforme avanza el horizonte de simulacion."
        ),
        (
            "2. Si la cola crece durante la simulacion, significa necesariamente que el modelo esta mal construido?",
            "<b>No.</b> De hecho, si con los datos de entrada (rho = 1.5625) la cola se mantuviera corta o vacia, "
            "el modelo estaria matematicamente equivocado. Una cola que crece de forma continua es el "
            "comportamiento fisico natural de cualquier sistema de colas donde la demanda supera a la oferta de servicio. "
            "El modelo refleja exactamente la realidad descrita por las tablas de frecuencias observadas."
        ),
        (
            "3. Que decision recomendarian: ampliar el horario, agilizar el servicio o habilitar otra estacion? Justifiquen con resultados.",
            f"Con rho = {rho_act:.4f}, se descarta <i>ampliar el horario</i> si la tasa horaria de demanda durante "
            "el periodo pico no cambia, pues la cola continuaria creciendo durante las horas operativas. "
            "La recomendacion primaria es <b>agilizar el servicio</b> si con cambios de procedimiento se puede llevar "
            "el servicio medio por debajo de 2.50 min (costo minimo). Si no es factible por limitaciones tecnicas de despacho, "
            "la decision categorica es <b>habilitar una segunda estacion de atencion</b>: con 2 servidores, "
            f"la capacidad asciende a 30 veh/h, reduciendo rho a {datos_base['lambda_hora'] / (datos_base['mu_hora']*2):.4f} (< 1), "
            "lo que garantizaria estabilidad estocastica, colas controladas y erradicacion del tiempo excesivo de espera."
        ),
    ]

    for preg_t, resp_t in preguntas_esc_def:
        st.markdown(
            f'<div class="crit-block">'
            f'<div class="crit-title">{preg_t}</div>'
            f'<div class="crit-a">{resp_t}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
