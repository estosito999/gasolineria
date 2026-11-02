import pandas as pd
import streamlit as st

from experimentos import ejecutar_replicas
from visualizacion import grafico_replicas


# ============================================================
# EJECUCIÓN DE RÉPLICAS
# ============================================================

def mostrar_instruccion5(n_replicas, n_clientes, semilla):
    """Muestra las réplicas y devuelve sus resultados para el análisis posterior."""

    # ========================================================
    # INSTRUCCIÓN 5
    # ========================================================

    st.header("🔁 Instrucción 5 — Ejecución de réplicas")

    st.info(
        "Ejecutar las réplicas. Registrar los resultados de cada ejecución "
        "con semillas aleatorias diferentes, cuando corresponda."
    )


    # ========================================================
    # 1. CONDICIONES OFICIALES
    # ========================================================

    st.subheader("⚙️ Condiciones de ejecución")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Réplicas oficiales", n_replicas)

    with col2:
        st.metric("Vehículos por réplica", n_clientes)

    with col3:
        st.metric("Semilla base", semilla)


    st.write(
        "**Una réplica representa una ejecución completa del modelo con "
        "100 vehículos.**"
    )

    st.latex(
        r"1\ \text{réplica}=100\ \text{vehículos}"
    )

    st.warning(
        "100 vehículos no significan 100 réplicas. "
        "La consigna establece 5 réplicas y cada una contiene 100 vehículos."
    )

    st.write(
        "Por lo tanto, se ejecutan cinco simulaciones independientes. "
        "En total se generan 500 vehículos, pero siguen siendo cinco "
        "ejecuciones separadas."
    )

    st.latex(
        r"5\ \text{réplicas}\times100\ \text{vehículos}=500\ \text{vehículos simulados}"
    )


    # ========================================================
    # 2. EJECUTAR LAS RÉPLICAS
    # ========================================================

    resultados_replicas, simulaciones_replicas = ejecutar_replicas(
        n_replicas,
        n_clientes,
        semilla
    )


    # ========================================================
    # 3. SEMILLAS DIFERENTES
    # ========================================================

    st.divider()

    st.subheader("🌱 Semillas reproducibles")

    st.write(
        "Cada réplica utiliza una semilla diferente. Esto permite que "
        "cada ejecución genere una secuencia distinta de números "
        "pseudoaleatorios."
    )

    st.write(
        "La primera réplica utiliza la semilla base. Para las siguientes "
        "réplicas se incrementa la semilla en 1009."
    )

    st.latex(
        r"s_r=s_{\text{base}}+(r-1)(1009),\qquad r=1,\ldots,5"
    )

    st.write(
        "Con una semilla base igual a 2026 obtenemos:"
    )

    semillas_df = resultados_replicas[
        ["replica", "semilla"]
    ].copy()

    semillas_df.columns = [
        "Réplica",
        "Semilla"
    ]

    st.dataframe(
        semillas_df,
        width="stretch",
        hide_index=True
    )

    st.caption(
        "En experimentos.py el ciclo comienza en r = 0, por eso internamente "
        "se utiliza semilla = semilla_base + r × 1009. Es equivalente a la "
        "fórmula mostrada arriba cuando numeramos las réplicas desde 1."
    )


    # ========================================================
    # 4. QUÉ CAMBIA Y QUÉ SE MANTIENE
    # ========================================================

    st.divider()

    st.subheader("🔄 ¿Qué ocurre en cada nueva réplica?")

    st.write(
        "Cada ejecución comienza nuevamente con el mismo modelo:"
    )

    condiciones = pd.DataFrame({
        "Elemento": [
            "Número de vehículos",
            "Número de servidores",
            "Disciplina",
            "Estado inicial",
            "Distribución de interarribos",
            "Distribución de servicio",
            "Indicador principal"
        ],
        "Se mantiene": [
            "100",
            "1 surtidor",
            "FIFO",
            "Servidor libre y cola vacía",
            "La misma distribución empírica",
            "La misma distribución empírica",
            "Tiempo medio de espera en cola"
        ]
    })

    st.dataframe(
        condiciones,
        width="stretch",
        hide_index=True
    )

    st.write(
        "Lo que cambia es la secuencia de números pseudoaleatorios debido "
        "a que cada réplica utiliza una semilla distinta."
    )

    st.code(
        """
MISMO MODELO
    ↓
Semilla diferente
    ↓
Números U diferentes
    ↓
Interarribos diferentes
    ↓
Servicios diferentes
    ↓
Esperas diferentes
    ↓
Resultado diferente para la réplica
        """,
        language=None
    )


    # ========================================================
    # 5. INDICADOR PRINCIPAL
    # ========================================================

    st.divider()

    st.subheader("⏳ Indicador principal de cada réplica")

    st.write(
        "La consigna establece como indicador principal el "
        "**tiempo medio de espera en cola**."
    )

    st.write(
        "En cada réplica tenemos 100 tiempos de espera:"
    )

    st.latex(
        r"W_{q,1},W_{q,2},\ldots,W_{q,100}"
    )

    st.write(
        "La media de espera de la réplica r se calcula como:"
    )

    st.latex(
        r"\bar{W}_{q,r}=\frac{1}{100}\sum_{i=1}^{100}W_{q,i}^{(r)}"
    )

    st.write(
        "Esto significa que sumamos los tiempos de espera de los "
        "100 vehículos de esa réplica y dividimos entre 100."
    )

    st.write(
        "También se incluyen los vehículos que tuvieron una espera "
        "igual a cero."
    )


    # ========================================================
    # 6. REGISTRO DE LAS CINCO EJECUCIONES
    # ========================================================

    st.divider()

    st.subheader("📋 Registro de las cinco ejecuciones")

    tabla_replicas = resultados_replicas[
        [
            "replica",
            "semilla",
            "n_clientes",
            "espera_media"
        ]
    ].copy()

    tabla_replicas.columns = [
        "Réplica",
        "Semilla utilizada",
        "Número de clientes",
        "Tiempo medio de espera en cola (min)"
    ]

    tabla_replicas[
        "Tiempo medio de espera en cola (min)"
    ] = tabla_replicas[
        "Tiempo medio de espera en cola (min)"
    ].round(2)

    st.dataframe(
        tabla_replicas,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 7. EXPLICACIÓN INDIVIDUAL DE LAS RÉPLICAS
    # ========================================================

    st.subheader("🔍 Resultado de cada réplica")

    for _, fila in tabla_replicas.iterrows():

        replica = int(fila["Réplica"])
        semilla_replica = int(fila["Semilla utilizada"])
        espera_replica = float(
            fila["Tiempo medio de espera en cola (min)"]
        )

        with st.expander(f"🔁 Réplica {replica}"):

            st.write(
                f"Semilla utilizada: **{semilla_replica}**"
            )

            st.write(
                "Número de vehículos simulados: **100**"
            )

            st.write(
                "Estado inicial: **servidor libre y cola vacía**"
            )

            st.write(
                f"Tiempo medio de espera en cola: "
                f"**{espera_replica:.2f} minutos**"
            )


    # ========================================================
    # 8. COMPARACIÓN GRÁFICA
    # ========================================================

    st.divider()

    st.subheader("📊 Comparación del tiempo medio de espera en cola")

    st.plotly_chart(
        grafico_replicas(resultados_replicas),
        width="stretch"
    )

    st.caption(
        "Cada punto representa el tiempo medio de espera obtenido en "
        "una ejecución completa de 100 vehículos."
    )


    # ========================================================
    # 9. MÉTRICAS COMPLEMENTARIAS
    # ========================================================

    with st.expander("📊 Ver resultados complementarios de las réplicas"):

        st.write(
            "experimentos.py también registra otros indicadores del sistema. "
            "Estos son complementarios; el indicador principal de la consigna "
            "continúa siendo el tiempo medio de espera en cola."
        )

        tabla_completa = resultados_replicas.copy()

        tabla_completa.columns = [
            "Réplica",
            "Semilla",
            "Clientes",
            "Espera media",
            "Espera máxima",
            "Cola máxima",
            "Tiempo medio en sistema",
            "Utilización observada",
            "Horizonte"
        ]

        st.dataframe(
            tabla_completa,
            width="stretch",
            hide_index=True
        )


    # ========================================================
    # 10. CONCLUSIÓN DE LA INSTRUCCIÓN
    # ========================================================

    st.divider()

    st.subheader("✅ Resultado de la Instrucción 5")

    st.success(
        "Se ejecutaron las 5 réplicas requeridas. Cada réplica simuló "
        "100 vehículos desde el estado inicial de servidor libre y cola "
        "vacía, utilizando una semilla diferente y reproducible. "
        "Para cada ejecución se registró el tiempo medio de espera en cola."
    )

    st.write(
        "Los cinco resultados obtenidos serán utilizados en la "
        "**Instrucción 6** para calcular la media global, el intervalo "
        "de confianza, el error relativo y el número requerido de réplicas."
    )

    return resultados_replicas, simulaciones_replicas
