import math

import streamlit as st

from datos import LLEGADAS, SERVICIO
from distribuciones import ajuste_poisson
from simulador import estado_en_tiempo
from visualizacion import grafico_timeline


# ============================================================
# ANÁLISIS DE DISTRIBUCIÓN Y SIMULACIÓN EMPÍRICA
# ============================================================

def mostrar_instruccion3(simulacion, resumen, n_clientes):

    st.header("📐 Instrucción 3 — Análisis de distribución")

    alpha_ajuste = 0.05

    st.info(
        "Realizar el análisis de distribución. Plantear una hipótesis de distribución, "
        "calcular las frecuencias esperadas y aplicar la prueba chi-cuadrado cuando se "
        "cumplan sus condiciones. Si no se ajusta una distribución teórica, se puede "
        "utilizar una distribución empírica discreta para simular los tiempos de "
        "interarribo y de servicio."
    )

    # ========================================================
    # PLANTEAMIENTO DEL ANÁLISIS
    # ========================================================

    st.subheader("🧪 Hipótesis de distribución")

    st.write(
        "Para analizar los tiempos de interarribo y de servicio proponemos una "
        "distribución de Poisson como modelo teórico candidato. La prueba de bondad "
        "de ajuste permitirá determinar si existe evidencia suficiente para rechazar "
        "esta hipótesis."
    )

    st.markdown(
        """
        **H₀:** Los datos observados son compatibles con la distribución Poisson propuesta.

        **H₁:** Los datos observados no son compatibles con la distribución Poisson propuesta.
        """
    )

    # ========================================================
    # NIVEL DE SIGNIFICANCIA
    # ========================================================

    st.subheader("🎯 Nivel de significancia")

    st.write(
        "Utilizaremos un nivel de significancia del 5 %. Después de calcular el "
        "p-valor, este valor será utilizado para tomar la decisión estadística."
    )

    st.latex(fr"\alpha = {alpha_ajuste}")

    st.latex(r"p < \alpha \Rightarrow \text{rechazar } H_0")

    st.latex(r"p \geq \alpha \Rightarrow \text{no rechazar } H_0")

    st.write(
        "A continuación realizaremos el análisis por separado para los tiempos "
        "de interarribo y los tiempos de servicio."
    )

    for nombre_datos, frecuencias in (
    ("A. Tiempos de interarribo", LLEGADAS),
    ("B. Tiempos de servicio", SERVICIO),
    ):

        st.divider()

        st.subheader(nombre_datos)

        ajuste = ajuste_poisson(
            frecuencias,
            alpha=alpha_ajuste
        )

        tabla_ajuste = ajuste["tabla"]


        # ========================================================
        # PARÁMETRO ESTIMADO
        # ========================================================

        st.markdown("### 🧮 Parámetro estimado de Poisson")

        st.write(
            "La distribución Poisson necesita un parámetro λ. "
            "En este análisis se estima utilizando la media ponderada "
            "de los mismos datos observados."
        )

        st.metric(
            "λ estimado con los mismos datos",
            f"{ajuste['lambda_estimado']:.4f}"
        )

        st.latex(
            r"\widehat{\lambda}"
            r"="
            r"\frac{\sum_x x f_x}{N}"
        )

        st.write(
            f"En este caso se analizaron **N = {ajuste['n']} observaciones**."
        )


        # ========================================================
        # CLASES ORIGINALES Y AGRUPACIÓN
        # ========================================================

        st.markdown("### 📦 Clases utilizadas en la prueba")

        st.write(
            "Para construir la prueba no utilizamos simplemente las cinco "
            "filas originales. La distribución Poisson también puede asignar "
            "probabilidad a valores que no aparecieron en la muestra."
        )

        st.write(
            f"Inicialmente se consideran **{ajuste['clases_originales']} categorías**, "
            f"incluyendo la cola de la distribución."
        )

        st.write(
            "Cuando alguna categoría tiene una frecuencia esperada menor que 5, "
            "se agrupa con una categoría contigua."
        )

        st.latex(r"E_i \geq 5")

        st.write(
            f"Después de realizar estas agrupaciones quedaron "
            f"**k = {len(tabla_ajuste)} clases finales**."
        )

        clases_finales = ", ".join(
            tabla_ajuste["clase"].astype(str).tolist()
        )

        st.write(
            f"Clases finales utilizadas: **{clases_finales}**"
        )

        st.caption(
            "El signo + indica categorías fusionadas y >= representa "
            "la cola final de la distribución."
        )


        # ========================================================
        # TABLA OBSERVADO VS ESPERADO
        # ========================================================

        st.markdown("### 📊 Frecuencias observadas y esperadas")

        st.write(
            "La frecuencia observada Oᵢ proviene de nuestros datos. "
            "La frecuencia esperada Eᵢ representa cuántos casos esperaríamos "
            "en esa clase si la distribución Poisson propuesta fuera compatible "
            "con los datos."
        )

        st.latex(r"E_i=N\,p_i")

        st.latex(
            r"p_i=P_{\widehat{\lambda}}(X\in C_i)"
        )

        st.dataframe(
            tabla_ajuste[
                [
                    "clase",
                    "observada",
                    "esperada",
                    "contribucion"
                ]
            ].rename(
                columns={
                    "clase": "Clase",
                    "observada": "Frecuencia observada O_i",
                    "esperada": "Frecuencia esperada E_i",
                    "contribucion": "(O_i - E_i)^2 / E_i",
                }
            ),
            width="stretch",
            hide_index=True
        )

        st.caption(
            f"Σ Oᵢ = {tabla_ajuste['observada'].sum():.0f}; "
            f"Σ Eᵢ = {tabla_ajuste['esperada'].sum():.4f}; "
            f"mínimo Eᵢ después de agrupar = "
            f"{tabla_ajuste['esperada'].min():.4f}."
        )


        # ========================================================
        # EJEMPLO DE UNA SOLA FILA
        # ========================================================

        st.markdown("### 🔍 Ejemplo paso a paso con una clase")

        # Preferimos una clase simple, por ejemplo "2" o "3",
        # para que el ejemplo sea más fácil de entender.
        clases_simples = tabla_ajuste[
            tabla_ajuste["clase"]
            .astype(str)
            .str.fullmatch(r"\d+")
        ]

        if not clases_simples.empty:

            fila_ejemplo = clases_simples.iloc[0]

        else:

            fila_ejemplo = tabla_ajuste.iloc[0]


        clase_ejemplo = str(
            fila_ejemplo["clase"]
        )

        observado_ejemplo = float(
            fila_ejemplo["observada"]
        )

        esperado_ejemplo = float(
            fila_ejemplo["esperada"]
        )

        contribucion_ejemplo = float(
            fila_ejemplo["contribucion"]
        )

        probabilidad_ejemplo = (
            esperado_ejemplo
            /
            ajuste["n"]
        )


        st.write(
            f"Tomemos como ejemplo la clase **{clase_ejemplo}**."
        )

        st.write(
            f"En nuestros datos observamos **Oᵢ = {observado_ejemplo:.0f}** casos."
        )

        st.write(
            "Según la distribución Poisson propuesta, la probabilidad "
            "teórica correspondiente a esa clase es aproximadamente:"
        )

        st.latex(
            fr"p_i"
            fr"="
            fr"\frac{{E_i}}{{N}}"
            fr"="
            fr"\frac{{{esperado_ejemplo:.4f}}}{{{ajuste['n']}}}"
            fr"="
            fr"{probabilidad_ejemplo:.6f}"
        )


        # ========================================================
        # SI LA CLASE ES UN ENTERO SIMPLE,
        # MOSTRAMOS TAMBIÉN LA FÓRMULA POISSON
        # ========================================================

        if clase_ejemplo.isdigit():

            x_ejemplo = int(
                clase_ejemplo
            )

            lambda_ejemplo = float(
                ajuste["lambda_estimado"]
            )

            st.write(
                "Como esta es una clase individual, su probabilidad "
                "proviene directamente de la función de masa de Poisson:"
            )

            st.latex(
                r"P(X=x)"
                r"="
                r"e^{-\widehat{\lambda}}"
                r"\frac{\widehat{\lambda}^{x}}{x!}"
            )

            st.latex(
                fr"P(X={x_ejemplo})"
                fr"="
                fr"e^{{-{lambda_ejemplo:.4f}}}"
                fr"\frac{{{lambda_ejemplo:.4f}^{{{x_ejemplo}}}}}"
                fr"{{{x_ejemplo}!}}"
                fr"\approx"
                fr"{probabilidad_ejemplo:.6f}"
            )


        # ========================================================
        # FRECUENCIA ESPERADA DEL EJEMPLO
        # ========================================================

        st.write(
            "Luego multiplicamos esa probabilidad por las "
            "50 observaciones para obtener la frecuencia esperada."
        )

        st.latex(
            r"E_i=Np_i"
        )

        st.latex(
            fr"E_i"
            fr"="
            fr"{ajuste['n']}"
            fr"({probabilidad_ejemplo:.6f})"
            fr"="
            fr"{esperado_ejemplo:.4f}"
        )

        st.write(
            f"Por tanto, observamos **{observado_ejemplo:.0f} casos**, "
            f"mientras que Poisson esperaba aproximadamente "
            f"**{esperado_ejemplo:.4f} casos**."
        )


        # ========================================================
        # CONTRIBUCIÓN AL CHI-CUADRADO
        # ========================================================

        st.write(
            "Ahora medimos la diferencia entre lo observado y lo esperado."
        )

        st.latex(
            r"\frac{(O_i-E_i)^2}{E_i}"
        )

        st.latex(
            fr"\frac{{"
            fr"({observado_ejemplo:.0f}"
            fr"-"
            fr"{esperado_ejemplo:.4f})^2"
            fr"}}"
            fr"{{{esperado_ejemplo:.4f}}}"
            fr"="
            fr"{contribucion_ejemplo:.6f}"
        )

        st.info(
            "Este valor todavía NO es el chi-cuadrado total. "
            "Es solamente la contribución de esta clase."
        )


        # ========================================================
        # CHI-CUADRADO TOTAL
        # ========================================================

        st.markdown("### 🧮 Cálculo del chi-cuadrado total")

        st.write(
            "Repetimos el mismo cálculo para todas las clases finales "
            "y sumamos todas sus contribuciones."
        )

        st.latex(
            r"\chi^2"
            r"="
            r"\sum_{i=1}^{k}"
            r"\frac{(O_i-E_i)^2}{E_i}"
        )

        if ajuste["valido"]:

            contribuciones_texto = " + ".join(
                f"{valor:.4f}"
                for valor in tabla_ajuste["contribucion"]
            )

            st.latex(
                fr"\chi^2"
                fr"="
                fr"{contribuciones_texto}"
                fr"="
                fr"{ajuste['chi2']:.6f}"
            )

            st.write(
                "Este es el estadístico chi-cuadrado calculado para "
                "comparar las frecuencias observadas con las esperadas."
            )


        # ========================================================
        # GRADOS DE LIBERTAD
        # ========================================================

        st.markdown("### 📐 Grados de libertad")

        st.write(
            "Los grados de libertad se calculan utilizando el número "
            "de clases FINALES después de realizar las agrupaciones."
        )

        st.latex(
            r"gl=k-1-q"
        )

        st.write(
            "donde:"
        )

        st.markdown(
            """
            - **k** = número de clases finales después de agrupar.
            - **1** = se descuenta porque el total de observaciones es fijo.
            - **q** = número de parámetros estimados con la misma muestra.
            """
        )

        st.write(
            "En nuestro ajuste Poisson estimamos solamente un parámetro: λ."
        )

        st.latex(r"q=1")

        k_final = len(
            tabla_ajuste
        )

        st.latex(
            fr"k={k_final}"
        )

        st.latex(
            fr"gl"
            fr"="
            fr"{k_final}-1-1"
        )

        if ajuste["gl"] is not None:

            st.latex(
                fr"\boxed{{gl={ajuste['gl']}}}"
            )

        st.warning(
            "k NO es automáticamente el número de filas de la tabla original. "
            "k representa el número de clases que quedan después de agrupar "
            "las categorías necesarias para cumplir las condiciones de la prueba."
        )


        # ========================================================
        # RESULTADOS PRINCIPALES
        # ========================================================

        st.markdown("### 📋 Resultado de la prueba")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "χ² calculado",
            f"{ajuste['chi2']:.6f}"
            if ajuste["valido"]
            else "No aplicable"
        )

        c2.metric(
            "Grados de libertad",
            ajuste["gl"]
        )

        c3.metric(
            "p-valor",
            f"{ajuste['pvalor']:.6g}"
            if ajuste["valido"]
            else "No aplicable"
        )

        c4.metric(
            "α",
            f"{ajuste['alpha']:.2f}"
        )


        # ========================================================
        # DE CHI-CUADRADO AL P-VALOR
        # ========================================================

        if ajuste["valido"]:

            st.markdown("### 🎯 Del chi-cuadrado al p-valor")

            st.write(
                "Una vez que conocemos el chi-cuadrado calculado y los "
                "grados de libertad, buscamos el área que queda a la derecha "
                "de ese valor en la distribución chi-cuadrado."
            )

            st.latex(
                fr"p"
                fr"="
                fr"P"
                fr"\left("
                fr"\chi^2_{{{ajuste['gl']}}}"
                fr"\geq"
                fr"{ajuste['chi2']:.6f}"
                fr"\right)"
            )

            st.latex(
                fr"p"
                fr"="
                fr"{ajuste['pvalor']:.6f}"
            )

            st.write(
                f"Esto equivale aproximadamente al "
                f"**{ajuste['pvalor'] * 100:.2f} %** del área "
                f"a la derecha de χ²."
            )


        # ========================================================
        # DECISIÓN ESTADÍSTICA
        # ========================================================

        st.markdown("### ⚖️ Decisión estadística")

        if not ajuste["valido"]:

            st.warning(
                f"No se aplica chi-cuadrado: {ajuste['motivo']} "
                "No se emite un p-valor ni una decisión de rechazo."
            )

        elif ajuste["rechazar"]:

            st.warning(
                f"p = {ajuste['pvalor']:.6g} < "
                f"α = {ajuste['alpha']:.2f} → rechazar H₀. "
                "Existe evidencia estadística suficiente para rechazar "
                "la compatibilidad con la distribución Poisson propuesta "
                "al nivel de significancia utilizado."
            )

        else:

            st.info(
                f"p = {ajuste['pvalor']:.6g} ≥ "
                f"α = {ajuste['alpha']:.2f} → no rechazar H₀. "
                "No existe evidencia estadística suficiente para rechazar "
                "H₀ al nivel de significancia utilizado. "
                "Esto NO demuestra que la distribución sea Poisson."
            )

    st.divider()
    st.subheader("Decisión para la simulación")
    st.latex(r"P_{\mathrm{emp}}(X=x_i)=\frac{f_i}{N}")
    st.info(
        "La simulación continuará utilizando la distribución empírica discreta "
        "derivada de las frecuencias observadas, tanto para interarribos como para "
        "servicios, mediante la transformada inversa de las Instrucciones 1 y 2. "
        "Esta distribución conserva las probabilidades y los valores registrados. "
        "No se dispone de una justificación matemática adicional para sustituirla "
        "por una distribución teórica: no rechazar H0 no basta para hacerlo. "
        "El análisis de Poisson solo evalúa el ajuste y no cambia el generador."
    )

    st.divider()
    st.subheader("Gasolinera — simulación con distribución empírica")


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
        width="stretch"
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
            width="stretch"
        )
