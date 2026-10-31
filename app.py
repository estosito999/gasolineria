import math

import streamlit as st


from datos import LLEGADAS
from datos import SERVICIO
from datos import obtener_distribuciones
from datos import resumen_datos

from generador import transformar_uniforme

from simulador import *

from experimentos import ejecutar_replicas

from estadistica import intervalo_confianza
from estadistica import replicas_requeridas

from distribuciones import ajuste_poisson

from visualizacion import *

from exportacion import *


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Gasolinera",
    page_icon="⛽",
    layout="wide"
)


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

datos_base = resumen_datos()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Configuración")


n_clientes = st.sidebar.number_input(
    "Vehículos por réplica",
    min_value=10,
    max_value=1000,
    value=100,
    step=10
)


semilla = st.sidebar.number_input(
    "Semilla",
    min_value=0,
    value=2026,
    step=1
)


n_replicas = st.sidebar.number_input(
    "Réplicas",
    min_value=2,
    max_value=100,
    value=5,
    step=1
)


confianza = st.sidebar.slider(
    "Nivel de confianza",
    min_value=0.80,
    max_value=0.99,
    value=0.95,
    step=0.01
)


precision = st.sidebar.slider(
    "Error relativo máximo",
    min_value=0.01,
    max_value=0.30,
    value=0.10,
    step=0.01
)


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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Datos",
    "🎲 Transformada",
    "⛽ Simulación",
    "🧪 Réplicas",
    "🧠 Conclusiones"
])


# ============================================================
# DATOS
# ============================================================

with tab1:

    st.header("Datos observados")


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Media entre llegadas",
        f"{datos_base['media_entre_llegadas']:.2f} min"
    )


    c2.metric(
        "Media servicio",
        f"{datos_base['media_servicio']:.2f} min"
    )


    c3.metric(
        "Llegadas",
        f"{datos_base['lambda_hora']:.2f} veh/h"
    )


    c4.metric(
        "Capacidad",
        f"{datos_base['mu_hora']:.2f} veh/h"
    )


    st.subheader("Carga del sistema")


    st.latex(
        r"\rho = \frac{\lambda}{\mu}"
    )


    st.latex(
        rf"\rho = {datos_base['rho']:.4f}"
    )


    if datos_base["rho"] > 1:

        st.error(
            "ρ > 1 → la demanda promedio supera "
            "la capacidad del surtidor."
        )


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "Tiempos entre llegadas"
        )

        st.dataframe(
            tabla_llegadas,
            use_container_width=True
        )


    with col2:

        st.subheader(
            "Tiempos de servicio"
        )

        st.dataframe(
            tabla_servicio,
            use_container_width=True
        )


    # --------------------------------------------------------
    # CHI CUADRADO
    # --------------------------------------------------------

    st.subheader(
        "Prueba exploratoria de Poisson"
    )


    llegada_poisson = ajuste_poisson(
        LLEGADAS
    )


    if llegada_poisson["valido"]:

        st.write(
            "χ²:",
            round(
                llegada_poisson["chi2"],
                4
            )
        )


        st.write(
            "p-valor:",
            round(
                llegada_poisson["pvalor"],
                5
            )
        )


        if llegada_poisson["rechazar"]:

            st.warning(
                "Se rechaza el ajuste Poisson. "
                "La simulación continuará usando "
                "la distribución empírica."
            )

        else:

            st.success(
                "No se rechaza el ajuste Poisson."
            )


        st.dataframe(
            llegada_poisson["tabla"],
            use_container_width=True
        )


# ============================================================
# TRANSFORMADA INVERSA
# ============================================================

with tab2:

    st.header(
        "Transformada inversa"
    )


    st.write(
        "Prueba qué ocurre con un número uniforme U."
    )


    u = st.slider(
        "U",
        0.0,
        0.9999,
        0.35,
        0.0001
    )


    tiempo_llegada = transformar_uniforme(
        u,
        tabla_llegadas
    )


    tiempo_servicio = transformar_uniforme(
        u,
        tabla_servicio
    )


    c1, c2 = st.columns(2)


    c1.metric(
        "Tiempo entre llegadas",
        f"{tiempo_llegada} min"
    )


    c2.metric(
        "Tiempo de servicio",
        f"{tiempo_servicio} min"
    )


    st.subheader(
        "Primeros cinco números aleatorios"
    )


    st.dataframe(
        simulacion[
            [
                "vehiculo",
                "u_llegada",
                "entre_llegadas",
                "u_servicio",
                "servicio"
            ]
        ].head(5),
        use_container_width=True
    )


# ============================================================
# SIMULACIÓN
# ============================================================

with tab3:

    st.header(
        "Gasolinera"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "Espera media",
        f"{resumen['espera_media']:.2f} min"
    )


    c2.metric(
        "Espera máxima",
        f"{resumen['espera_maxima']:.2f} min"
    )


    c3.metric(
        "Cola máxima",
        resumen["cola_maxima"]
    )


    c4.metric(
        "Utilización",
        f"{resumen['utilizacion_observada'] * 100:.1f}%"
    )


    # --------------------------------------------------------
    # TIMELINE
    # --------------------------------------------------------

    st.plotly_chart(
        grafico_timeline(
            simulacion,
            min(30, int(n_clientes))
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # RELOJ INTERACTIVO
    # --------------------------------------------------------

    st.subheader(
        "Reloj de simulación"
    )


    t = st.slider(
        "Tiempo",
        min_value=0,
        max_value=int(
            math.ceil(
                resumen["horizonte"]
            )
        ),
        value=0,
        step=1
    )


    estado = estado_en_tiempo(
        simulacion,
        t
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Esperando",
        len(
            estado["esperando"]
        )
    )


    if estado["atendiendo"]:

        servidor = (
            f"V{estado['atendiendo'][0]}"
        )

    else:

        servidor = "LIBRE"


    c2.metric(
        "Surtidor",
        servidor
    )


    c3.metric(
        "Finalizados",
        len(
            estado["finalizados"]
        )
    )


    # --------------------------------------------------------
    # COLA FIFO
    # --------------------------------------------------------

    st.subheader(
        "Cola FIFO"
    )


    if estado["esperando"]:

        texto = " → ".join(
            f"V{x}"
            for x in estado["esperando"]
        )

        st.write(texto)

    else:

        st.success(
            "Cola vacía"
        )


    # --------------------------------------------------------
    # INSPECCIONAR VEHÍCULO
    # --------------------------------------------------------

    st.subheader(
        "Inspeccionar vehículo"
    )


    vehiculo = st.number_input(
        "Vehículo",
        min_value=1,
        max_value=int(n_clientes),
        value=1
    )


    fila = simulacion.iloc[
        int(vehiculo) - 1
    ]


    st.latex(
        r"B_i=\max(T_i,D_{i-1})"
    )


    st.latex(
        r"W_{q,i}=B_i-T_i"
    )


    st.write({

        "llegada":
            int(fila["llegada"]),

        "servicio":
            int(fila["servicio"]),

        "inicio":
            int(fila["inicio_servicio"]),

        "salida":
            int(fila["salida"]),

        "espera":
            int(fila["espera_cola"])
    })


    # --------------------------------------------------------
    # TABLA COMPLETA
    # --------------------------------------------------------

    with st.expander(
        "Ver tabla completa"
    ):

        st.dataframe(
            simulacion,
            use_container_width=True
        )


# ============================================================
# RÉPLICAS
# ============================================================

with tab4:

    st.header(
        "Réplicas independientes"
    )


    resultados, simulaciones = ejecutar_replicas(
        int(n_replicas),
        int(n_clientes),
        int(semilla)
    )


    st.dataframe(
        resultados,
        use_container_width=True
    )


    st.plotly_chart(
        grafico_replicas(
            resultados
        ),
        use_container_width=True
    )


    # --------------------------------------------------------
    # INTERVALO DE CONFIANZA
    # --------------------------------------------------------

    ic = intervalo_confianza(
        resultados["espera_media"],
        confianza
    )


    st.subheader(
        "Intervalo de confianza"
    )


    st.latex(
        r"""
        \bar{Y}
        \pm
        t_{\alpha/2,R-1}
        \frac{s_Y}{\sqrt{R}}
        """
    )


    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Media",
        f"{ic['media']:.2f}"
    )


    c2.metric(
        "Intervalo",
        f"[{ic['li']:.2f}, {ic['ls']:.2f}]"
    )


    c3.metric(
        "Error relativo",
        f"{ic['error_relativo'] * 100:.2f}%"
    )


    # --------------------------------------------------------
    # RÉPLICAS NECESARIAS
    # --------------------------------------------------------

    requeridas = replicas_requeridas(
        ic["media"],
        ic["desviacion"],
        precision,
        confianza
    )


    st.metric(
        "Réplicas estimadas necesarias",
        requeridas
    )


    if ic["error_relativo"] <= precision:

        st.success(
            "La precisión solicitada se cumple."
        )

    else:

        st.warning(
            "Todavía se necesitan más réplicas."
        )


# ============================================================
# CONCLUSIONES
# ============================================================

with tab5:

    st.header(
        "Interpretación"
    )


    st.subheader(
        "¿La demanda supera la capacidad?"
    )


    st.write(
        f"""
        El tiempo medio entre llegadas es
        {datos_base['media_entre_llegadas']:.2f} minutos.

        El tiempo medio de servicio es
        {datos_base['media_servicio']:.2f} minutos.
        """
    )


    st.latex(
        rf"""
        \rho =
        {datos_base['rho']:.4f}
        """
    )


    if datos_base["rho"] > 1:

        st.error(
            "La demanda promedio supera "
            "la capacidad del único surtidor."
        )


    st.subheader(
        "¿Una cola creciente significa que el programa está mal?"
    )


    st.write(
        """
        No necesariamente.

        En este escenario llegan vehículos
        más rápido de lo que el surtidor
        puede atenderlos.

        Por tanto una cola creciente es
        compatible con los datos observados.
        """
    )


    st.subheader(
        "Escenarios futuros"
    )


    st.write(
        """
        En la siguiente etapa podremos comparar:

        • un surtidor actual

        • servicio más rápido

        • dos surtidores

        • incremento de capacidad en hora pico
        """
    )