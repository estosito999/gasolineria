import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import t as student_t


def mostrar_instruccion6(resultados_replicas, confianza, precision):
    """Analiza las mismas medias de las réplicas mostradas en la Instrucción 5."""

    # ========================================================
    # INSTRUCCIÓN 6
    # ========================================================

    st.header("📈 Instrucción 6 — Análisis estadístico de los resultados")

    st.info(
        "Analizar estadísticamente los resultados. Calcular la media de espera, "
        "su intervalo de confianza, el error relativo y el número de réplicas "
        "requerido según la precisión solicitada."
    )


    # ========================================================
    # 1. RESULTADOS QUE VIENEN DE LA INSTRUCCIÓN 5
    # ========================================================

    st.subheader("🔁 Resultados obtenidos en las réplicas")

    st.write(
        "En la Instrucción 5 ejecutamos cinco réplicas independientes. "
        "Cada réplica produjo un tiempo medio de espera en cola."
    )

    st.latex(
        r"Y_r=\text{tiempo medio de espera en cola de la réplica }r"
    )

    valores = resultados_replicas[
        "espera_media"
    ].astype(float).to_numpy()

    R = len(valores)

    tabla_valores = pd.DataFrame({
        "Réplica": range(1, R + 1),
        "Tiempo medio de espera en cola (min)": valores
    })

    tabla_valores[
        "Tiempo medio de espera en cola (min)"
    ] = tabla_valores[
        "Tiempo medio de espera en cola (min)"
    ].round(2)

    st.dataframe(
        tabla_valores,
        width="stretch",
        hide_index=True
    )

    st.write(
        "Por lo tanto, nuestros datos estadísticos ya no son los "
        "100 vehículos individuales. Ahora trabajamos con las medias "
        "obtenidas en las cinco réplicas."
    )

    st.latex(
        r"Y_1,Y_2,Y_3,Y_4,Y_5"
    )


    # ========================================================
    # 2. MEDIA GLOBAL
    # ========================================================

    st.divider()

    st.subheader("🧮 1. Media de espera")

    st.write(
        "Primero calculamos la media de los resultados obtenidos "
        "en las réplicas."
    )

    st.latex(
        r"\bar{Y}=\frac{1}{R}\sum_{r=1}^{R}Y_r"
    )

    media_global = float(
        np.mean(valores)
    )

    st.write(
        f"En nuestro caso tenemos **R = {R} réplicas**."
    )

    expresion_valores = " + ".join(
        f"{valor:.2f}"
        for valor in valores
    )

    st.latex(
        fr"\bar{{Y}}=\frac{{{expresion_valores}}}{{{R}}}"
    )

    st.latex(
        fr"\bar{{Y}}={media_global:.2f}\ \text{{min}}"
    )

    st.metric(
        "⏳ Media global de espera en cola",
        f"{media_global:.2f} min"
    )

    st.success(
        f"En promedio, considerando las {R} réplicas, un vehículo "
        f"esperó aproximadamente {media_global:.2f} minutos en cola."
    )


    # ========================================================
    # 3. DESVIACIÓN ESTÁNDAR DE LAS RÉPLICAS
    # ========================================================

    st.divider()

    st.subheader("📊 2. Variabilidad entre las réplicas")

    st.write(
        "Las cinco réplicas no producen exactamente el mismo resultado. "
        "Necesitamos medir cuánto varían sus medias."
    )

    st.latex(
        r"s_Y=\sqrt{\frac{\sum_{r=1}^{R}(Y_r-\bar{Y})^2}{R-1}}"
    )

    desviacion = float(
        np.std(
            valores,
            ddof=1
        )
    )

    st.metric(
        "Desviación estándar entre réplicas",
        f"{desviacion:.2f} min"
    )

    st.write(
        "Una desviación mayor indica que las distintas ejecuciones "
        "del modelo están produciendo resultados más dispersos."
    )


    # ========================================================
    # 4. ERROR ESTÁNDAR
    # ========================================================

    st.divider()

    st.subheader("📏 3. Error estándar de la media")

    st.write(
        "La desviación estándar mide la variabilidad de las réplicas. "
        "Pero para construir el intervalo de confianza necesitamos "
        "el error estándar de la media."
    )

    st.latex(
        r"SE=\frac{s_Y}{\sqrt{R}}"
    )

    error_estandar = desviacion / math.sqrt(R)

    st.latex(
        fr"SE=\frac{{{desviacion:.4f}}}{{\sqrt{{{R}}}}}"
    )

    st.latex(
        fr"SE={error_estandar:.4f}\ \text{{min}}"
    )


    # ========================================================
    # 5. DISTRIBUCIÓN t DE STUDENT
    # ========================================================

    st.divider()

    st.subheader("🎯 4. Valor crítico t de Student")

    st.write(
        "Como solamente tenemos cinco réplicas y la desviación estándar "
        "poblacional es desconocida, utilizamos la distribución "
        "t de Student."
    )

    alpha = 1 - confianza

    grados_libertad = R - 1

    t_critico = float(
        student_t.ppf(
            1 - alpha / 2,
            grados_libertad
        )
    )

    st.latex(
        r"gl=R-1"
    )

    st.latex(
        fr"gl={R}-1={grados_libertad}"
    )

    st.write(
        f"Para un nivel de confianza del {confianza * 100:.0f} % "
        f"y {grados_libertad} grados de libertad:"
    )

    st.latex(
        fr"t_{{0.975,{grados_libertad}}}={t_critico:.4f}"
    )


    # ========================================================
    # 6. SEMIAMPLITUD
    # ========================================================

    st.divider()

    st.subheader("📐 5. Semiamplitud del intervalo")

    st.write(
        "La semiamplitud indica cuánto se extiende el intervalo "
        "hacia cada lado de la media."
    )

    st.latex(
        r"h=t_{\alpha/2,R-1}\frac{s_Y}{\sqrt{R}}"
    )

    h = t_critico * error_estandar

    st.latex(
        fr"h={t_critico:.4f}\left({error_estandar:.4f}\right)"
    )

    st.latex(
        fr"h={h:.2f}\ \text{{min}}"
    )

    st.metric(
        "Semiamplitud",
        f"{h:.2f} min"
    )


    # ========================================================
    # 7. INTERVALO DE CONFIANZA
    # ========================================================

    st.divider()

    st.subheader("📊 6. Intervalo de confianza del 95 %")

    st.write(
        "El intervalo de confianza se construye alrededor de la "
        "media global."
    )

    st.latex(
        r"IC_{95\%}=\bar{Y}\pm h"
    )

    limite_inferior = media_global - h

    limite_superior = media_global + h

    st.latex(
        fr"IC_{{95\%}}={media_global:.2f}\pm{h:.2f}"
    )

    st.latex(
        fr"IC_{{95\%}}="
        fr"\left[{limite_inferior:.2f},"
        fr"{limite_superior:.2f}\right]"
        fr"\ \text{{min}}"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Límite inferior",
            f"{limite_inferior:.2f} min"
        )

    with col2:

        st.metric(
            "Media",
            f"{media_global:.2f} min"
        )

    with col3:

        st.metric(
            "Límite superior",
            f"{limite_superior:.2f} min"
        )

    st.write(
        "Este intervalo representa la incertidumbre estadística "
        "asociada a nuestra estimación del tiempo medio de espera."
    )


    # ========================================================
    # 8. ERROR RELATIVO
    # ========================================================

    st.divider()

    st.subheader("🎯 7. Error relativo")

    st.write(
        "La consigna establece que la precisión deseada corresponde "
        "a un error relativo máximo del 10 %."
    )

    st.latex(
        r"e_r=\frac{h}{|\bar{Y}|}"
    )

    if media_global != 0:

        error_relativo = h / abs(media_global)

    else:

        error_relativo = float("inf")


    error_porcentaje = (
        error_relativo * 100
    )

    st.latex(
        fr"e_r="
        fr"\frac{{{h:.2f}}}{{{media_global:.2f}}}"
        fr"={error_relativo:.4f}"
    )

    st.latex(
        fr"e_r={error_porcentaje:.2f}\%"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Error relativo obtenido",
            f"{error_porcentaje:.2f} %"
        )

    with col2:

        st.metric(
            "Error máximo permitido",
            f"{precision * 100:.0f} %"
        )


    # ========================================================
    # 9. VERIFICAR LA PRECISIÓN
    # ========================================================

    cumple_precision = (
        error_relativo <= precision
    )

    if cumple_precision:

        st.success(
            "✅ El número actual de réplicas cumple la precisión solicitada."
        )

    else:

        st.warning(
            "⚠️ Las 5 réplicas iniciales todavía no permiten alcanzar "
            "el error relativo máximo del 10 %. Se debe estimar cuántas "
            "réplicas serían necesarias."
        )


    # ========================================================
    # 10. NÚMERO DE RÉPLICAS REQUERIDO
    # ========================================================

    st.divider()

    st.subheader("🔢 8. Número de réplicas requerido")

    st.write(
        "Utilizamos la variabilidad observada en las réplicas iniciales "
        "para estimar cuántas ejecuciones serían necesarias para alcanzar "
        "la precisión solicitada."
    )

    st.latex(
        r"""
        R^*
        \approx
        \left(
        \frac{
        t\,s_Y
        }{
        e_{\max}|\bar{Y}|
        }
        \right)^2
        """
    )

    st.write(
        "Como el valor crítico t depende de los grados de libertad, "
        "el cálculo se realiza iterativamente."
    )


    # ========================================================
    # CÁLCULO ITERATIVO
    # ========================================================

    replicas_requeridas = max(
        R,
        2
    )

    for _ in range(100):

        t_requerido = float(
            student_t.ppf(
                1 - alpha / 2,
                replicas_requeridas - 1
            )
        )

        nueva_estimacion = math.ceil(
            (
                (
                    t_requerido
                    *
                    desviacion
                )
                /
                (
                    precision
                    *
                    abs(media_global)
                )
            ) ** 2
        )

        nueva_estimacion = max(
            R,
            nueva_estimacion
        )

        if nueva_estimacion == replicas_requeridas:
            break

        replicas_requeridas = nueva_estimacion


    replicas_adicionales = max(
        0,
        replicas_requeridas - R
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Réplicas totales estimadas",
            replicas_requeridas
        )

    with col2:

        st.metric(
            "Réplicas adicionales estimadas",
            replicas_adicionales
        )


    if replicas_adicionales > 0:

        st.warning(
            f"Con la variabilidad observada en las {R} réplicas iniciales, "
            f"se estiman aproximadamente {replicas_requeridas} réplicas "
            f"totales para alcanzar un error relativo máximo del "
            f"{precision * 100:.0f} %. Esto implica ejecutar unas "
            f"{replicas_adicionales} réplicas adicionales."
        )

    else:

        st.success(
            "Las réplicas actuales ya son suficientes para alcanzar "
            "la precisión solicitada."
        )


    # ========================================================
    # 11. RESUMEN ESTADÍSTICO
    # ========================================================

    st.divider()

    st.subheader("📋 Resumen estadístico")

    resumen_estadistico = pd.DataFrame({
        "Indicador": [
            "Número de réplicas iniciales",
            "Media global de espera",
            "Desviación estándar",
            "Error estándar",
            "Valor crítico t",
            "Semiamplitud",
            "Límite inferior IC",
            "Límite superior IC",
            "Error relativo",
            "Precisión máxima permitida",
            "Réplicas totales estimadas"
        ],

        "Resultado": [
            str(R),
            f"{media_global:.2f} min",
            f"{desviacion:.2f} min",
            f"{error_estandar:.2f} min",
            f"{t_critico:.4f}",
            f"{h:.2f} min",
            f"{limite_inferior:.2f} min",
            f"{limite_superior:.2f} min",
            f"{error_porcentaje:.2f} %",
            f"{precision * 100:.0f} %",
            str(replicas_requeridas)
        ]
    })

    st.dataframe(
        resumen_estadistico,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 12. CONCLUSIÓN
    # ========================================================

    st.divider()

    st.subheader("✅ Resultado de la Instrucción 6")

    st.write(
        f"Las {R} réplicas iniciales produjeron un tiempo medio de "
        f"espera global de **{media_global:.2f} minutos**."
    )

    st.write(
        f"Con un nivel de confianza del {confianza * 100:.0f} %, "
        f"el intervalo estimado es de **{limite_inferior:.2f} a "
        f"{limite_superior:.2f} minutos**."
    )

    if cumple_precision:

        st.success(
            f"El error relativo obtenido es {error_porcentaje:.2f} %, "
            f"por lo que cumple con la precisión máxima solicitada "
            f"del {precision * 100:.0f} %."
        )

    else:

        st.warning(
            f"El error relativo obtenido es {error_porcentaje:.2f} %, "
            f"superior al máximo solicitado del {precision * 100:.0f} %. "
            f"Por ello se estiman aproximadamente "
            f"{replicas_requeridas} réplicas totales."
        )
