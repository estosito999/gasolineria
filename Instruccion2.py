import numpy as np
import pandas as pd
import streamlit as st

from generador import transformar_uniforme


# ============================================================
# TRANSFORMADA INVERSA
# ============================================================

def mostrar_instruccion2(tabla_llegadas, tabla_servicio, n_clientes, semilla):

    # ========================================================
    # INSTRUCCIÓN 2
    # ========================================================

    st.header("🎲 Instrucción 2 — Generación de variables aleatorias")

    st.info(
        "Construir una distribución acumulada y aplicar la transformada "
        "inversa utilizando números aleatorios uniformes entre 0 y 1."
    )


    # ========================================================
    # 1. NÚMEROS UNIFORMES
    # ========================================================

    st.subheader("🎯 Números aleatorios uniformes")

    st.write(
        "La simulación utiliza números pseudoaleatorios uniformes "
        "comprendidos entre 0 y 1."
    )

    st.latex(r"U \sim U(0,1)")

    st.write(
        "Cada número U será transformado en un tiempo de interarribo "
        "o en un tiempo de servicio según la distribución acumulada."
    )

    st.latex(r"U_i \longrightarrow A_i")

    st.latex(r"U_i \longrightarrow S_i")


    # ========================================================
    # 2. DISTRIBUCIÓN ACUMULADA DE INTERARRIBOS
    # ========================================================

    st.divider()

    st.subheader("🚗 A. Distribución acumulada de los interarribos")

    st.write(
        "Partimos de las probabilidades obtenidas en la Instrucción 1 "
        "y las acumulamos progresivamente."
    )

    st.latex(r"F_i = \sum_{j=1}^{i} p_j")

    tabla_interarribos = tabla_llegadas[
        [
            "tiempo",
            "frecuencia",
            "probabilidad",
            "acumulada",
            "limite_inferior",
            "limite_superior"
        ]
    ].copy()

    tabla_interarribos["Intervalo de U"] = tabla_interarribos.apply(
        lambda fila: f"[{fila['limite_inferior']:.2f}, {fila['limite_superior']:.2f})",
        axis=1
    )

    tabla_interarribos.columns = [
        "Interarribo (min)",
        "Frecuencia",
        "Probabilidad",
        "Acumulada",
        "Límite inferior",
        "Límite superior",
        "Intervalo de U"
    ]

    st.dataframe(
        tabla_interarribos,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # EXPLICACIÓN DE LA ACUMULADA
    # ========================================================

    st.markdown("### 🧮 ¿Cómo se construye la acumulada?")

    st.write("Para un interarribo de 1 minuto:")

    st.latex(r"F(1)=0.20")

    st.write("Para un interarribo de 2 minutos:")

    st.latex(r"F(2)=0.20+0.30=0.50")

    st.write("Para un interarribo de 3 minutos:")

    st.latex(r"F(3)=0.50+0.30=0.80")

    st.write(
        "El procedimiento continúa hasta alcanzar una probabilidad "
        "acumulada igual a 1."
    )

    st.latex(r"F_{\text{final}}=1")


    # ========================================================
    # 3. DISTRIBUCIÓN ACUMULADA DEL SERVICIO
    # ========================================================

    st.divider()

    st.subheader("⛽ B. Distribución acumulada de los tiempos de servicio")

    st.write(
        "Aplicamos exactamente el mismo procedimiento a los tiempos "
        "de servicio observados."
    )

    tabla_servicios = tabla_servicio[
        [
            "tiempo",
            "frecuencia",
            "probabilidad",
            "acumulada",
            "limite_inferior",
            "limite_superior"
        ]
    ].copy()

    tabla_servicios["Intervalo de U"] = tabla_servicios.apply(
        lambda fila: f"[{fila['limite_inferior']:.2f}, {fila['limite_superior']:.2f})",
        axis=1
    )

    tabla_servicios.columns = [
        "Servicio (min)",
        "Frecuencia",
        "Probabilidad",
        "Acumulada",
        "Límite inferior",
        "Límite superior",
        "Intervalo de U"
    ]

    st.dataframe(
        tabla_servicios,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 4. TRANSFORMADA INVERSA
    # ========================================================

    st.divider()

    st.subheader("🔄 Transformada inversa")

    st.write(
        "Una vez construidos los intervalos, generamos un número U "
        "entre 0 y 1 y observamos en qué intervalo se encuentra."
    )

    st.latex(r"F_{j-1} \leq U < F_j")

    st.write(
        "El intervalo donde cae U determina el valor de la variable "
        "aleatoria generada."
    )

    st.markdown("**Para los interarribos:**")

    st.latex(r"""
    A =
    \begin{cases}
    1 & 0.00 \leq U < 0.20 \\
    2 & 0.20 \leq U < 0.50 \\
    3 & 0.50 \leq U < 0.80 \\
    4 & 0.80 \leq U < 0.94 \\
    5 & 0.94 \leq U < 1.00
    \end{cases}
    """)

    st.markdown("**Para los tiempos de servicio:**")

    st.latex(r"""
    S =
    \begin{cases}
    2 & 0.00 \leq U < 0.10 \\
    3 & 0.10 \leq U < 0.30 \\
    4 & 0.30 \leq U < 0.70 \\
    5 & 0.70 \leq U < 0.90 \\
    6 & 0.90 \leq U < 1.00
    \end{cases}
    """)


    # ========================================================
    # 5. GENERAR LOS NÚMEROS ALEATORIOS
    # ========================================================

    st.divider()

    st.subheader("🔢 Primeros cinco números aleatorios utilizados")

    rng = np.random.default_rng(semilla)

    u_llegadas = rng.random(n_clientes)
    u_servicios = rng.random(n_clientes)

    interarribos_generados = [
        transformar_uniforme(u, tabla_llegadas)
        for u in u_llegadas
    ]

    servicios_generados = [
        transformar_uniforme(u, tabla_servicio)
        for u in u_servicios
    ]

    evidencia = pd.DataFrame({
        "Vehículo": range(1, 6),
        "U interarribo": [round(u, 5) for u in u_llegadas[:5]],
        "Interarribo (min)": interarribos_generados[:5],
        "U servicio": [round(u, 5) for u in u_servicios[:5]],
        "Servicio (min)": servicios_generados[:5]
    })

    st.dataframe(
        evidencia,
        width="stretch",
        hide_index=True
    )

    st.caption(
        f"Los números anteriores fueron generados utilizando la semilla {semilla}."
    )

    st.markdown("### 🧠 Interpretación de la tabla")

    st.write(
        "Cada vehículo utiliza dos números pseudoaleatorios uniformes entre 0 y 1: "
        "uno para determinar su tiempo de interarribo y otro para determinar "
        "su tiempo de servicio."
    )

    st.write(
        "Los valores U no representan minutos directamente. Su función es "
        "seleccionar un intervalo dentro de la distribución acumulada. "
        "El intervalo encontrado determina el tiempo generado."
    )

    st.markdown("#### 📖 ¿Cómo leer cada fila?")

    st.write(
        """
        - **Vehículo:** identifica al vehículo simulado.
        - **U interarribo:** número uniforme utilizado para generar el interarribo.
        - **Interarribo (min):** tiempo obtenido mediante la transformada inversa.
        - **U servicio:** número uniforme utilizado para generar el servicio.
        - **Servicio (min):** tiempo de servicio obtenido mediante la transformada inversa.
        """
    )

    for _, fila in evidencia.iterrows():

        vehiculo = int(fila["Vehículo"])
        u_interarribo = float(fila["U interarribo"])
        interarribo = int(fila["Interarribo (min)"])
        u_servicio = float(fila["U servicio"])
        servicio = int(fila["Servicio (min)"])

        st.write(
            f"- Para el **vehículo {vehiculo}**, el número "
            f"**U = {u_interarribo:.5f}** produjo un interarribo de "
            f"**{interarribo} minutos**. Por otra parte, el número "
            f"**U = {u_servicio:.5f}** produjo un tiempo de servicio de "
            f"**{servicio} minutos**."
        )

    st.info(
        "En resumen: U no representa directamente un tiempo. "
        "El número U indica qué intervalo de la distribución acumulada debe "
        "seleccionarse y ese intervalo determina el tiempo generado."
    )

    # ========================================================
    # 6. EJEMPLO DE TRANSFORMACIÓN DE INTERARRIBO
    # ========================================================

    st.divider()

    st.subheader("🧮 Ejemplo 1 — Transformación de un interarribo")

    u_a = float(u_llegadas[0])
    a = interarribos_generados[0]

    fila_a = tabla_llegadas[
        tabla_llegadas["tiempo"] == a
    ].iloc[0]

    li_a = float(fila_a["limite_inferior"])
    ls_a = float(fila_a["limite_superior"])

    st.write(
        f"Para el primer vehículo se generó el número uniforme U = {u_a:.5f}."
    )

    st.write(
        "Ahora buscamos el intervalo de la distribución acumulada "
        "donde se encuentra este número."
    )

    st.latex(
        fr"{li_a:.2f} \leq {u_a:.5f} < {ls_a:.2f}"
    )

    st.write(
        f"Por lo tanto, el número pertenece al intervalo correspondiente "
        f"a un interarribo de {a} minutos."
    )

    st.latex(
        fr"A_1={a}\ \text{{min}}"
    )


    # ========================================================
    # 7. EJEMPLO DE TRANSFORMACIÓN DE SERVICIO
    # ========================================================

    st.subheader("🧮 Ejemplo 2 — Transformación de un tiempo de servicio")

    u_s = float(u_servicios[0])
    s = servicios_generados[0]

    fila_s = tabla_servicio[
        tabla_servicio["tiempo"] == s
    ].iloc[0]

    li_s = float(fila_s["limite_inferior"])
    ls_s = float(fila_s["limite_superior"])

    st.write(
        f"Para el primer servicio se generó el número uniforme U = {u_s:.5f}."
    )

    st.write(
        "Buscamos nuevamente el intervalo de la distribución acumulada "
        "donde se encuentra este número."
    )

    st.latex(
        fr"{li_s:.2f} \leq {u_s:.5f} < {ls_s:.2f}"
    )

    st.write(
        f"Por lo tanto, el número pertenece al intervalo correspondiente "
        f"a un tiempo de servicio de {s} minutos."
    )

    st.latex(
        fr"S_1={s}\ \text{{min}}"
    )


    # ========================================================
    # 8. FLUJO DEL PROCEDIMIENTO
    # ========================================================

    st.divider()

    st.subheader("🧠 Proceso completo")

    st.code(
        """
Datos observados
        ↓
Frecuencia relativa
        ↓
Distribución acumulada
        ↓
Generar U entre 0 y 1
        ↓
Buscar el intervalo donde cae U
        ↓
Aplicar transformada inversa
        ↓
Obtener interarribo o tiempo de servicio
        """,
        language=None
    )

    st.latex(
        r"f_i \longrightarrow p_i \longrightarrow F_i \longrightarrow U_i \longrightarrow X_i"
    )


    # ========================================================
    # 9. DEFENSA DEL EXAMEN
    # ========================================================

    st.divider()

    with st.expander("🎓 ¿Cómo defender esta instrucción en el examen?"):

        st.markdown("**1. ¿Qué números aleatorios utilizamos?**")

        st.write(
            "Utilizamos números pseudoaleatorios uniformes "
            "comprendidos entre 0 y 1."
        )

        st.latex(r"U\sim U(0,1)")


        st.markdown("**2. ¿Qué es la distribución acumulada?**")

        st.write(
            "Es la suma progresiva de las probabilidades de cada valor."
        )

        st.latex(r"F_i=\sum_{j=1}^{i}p_j")


        st.markdown("**3. ¿Para qué sirve la distribución acumulada?**")

        st.write(
            "Permite dividir el intervalo entre 0 y 1 en regiones "
            "proporcionales a las probabilidades observadas."
        )


        st.markdown("**4. ¿Cómo funciona la transformada inversa?**")

        st.write(
            "Se genera un número U y se identifica el intervalo "
            "de la distribución acumulada donde se encuentra."
        )

        st.latex(r"F_{j-1}\leq U<F_j")


        st.markdown("**5. ¿Qué obtenemos después?**")

        st.write(
            "Obtenemos un tiempo de interarribo o un tiempo de servicio "
            "que sigue la distribución empírica construida a partir "
            "de los datos observados."
        )


        st.markdown("**6. ¿Por qué usamos una semilla?**")

        st.write(
            "La semilla permite reproducir la misma secuencia de números "
            "pseudoaleatorios y verificar nuevamente los resultados."
        )
