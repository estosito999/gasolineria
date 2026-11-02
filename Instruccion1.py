import streamlit as st


# ============================================================
# DATOS
# ============================================================
def mostrar_instruccion1(tabla_llegadas, tabla_servicio):

    st.header("📊 Instrucción 1 — Análisis de los datos observados")

    st.info(
        "Analizar los datos observados. Calcular las medias y frecuencias "
        "relativas de los tiempos entre llegadas y de servicio."
    )

    st.write(
        """
        En esta sección analizamos las **50 observaciones de tiempos de interarribo**
        y las **50 observaciones de tiempos de servicio** proporcionadas en el ejercicio.
        """
    )

    # ========================================================
    # A. TIEMPOS DE INTERARRIBO
    # ========================================================

    st.divider()

    st.subheader("🚗 A. Tiempos de interarribo")

    st.write(
        """
        El **tiempo de interarribo** representa el tiempo transcurrido
        entre la llegada de un vehículo y la llegada del siguiente.
        """
    )

    datos_interarribo = tabla_llegadas[
        ["tiempo", "frecuencia", "probabilidad"]
    ].copy()

    datos_interarribo.columns = [
        "Tiempo de interarribo (min)",
        "Frecuencia",
        "Frecuencia relativa"
    ]

    datos_interarribo["Porcentaje"] = (
        datos_interarribo["Frecuencia relativa"] * 100
    )

    st.dataframe(
        datos_interarribo,
        width="stretch",
        hide_index=True
    )
    
    # ========================================================
    # INTERPRETACIÓN DE LA TABLA
    # ========================================================

    st.markdown("### 🧠 Interpretación de la tabla")

    st.write(
        "La frecuencia indica cuántas veces se observó cada tiempo "
        "de interarribo dentro de las 50 observaciones."
    )

    for _, fila in datos_interarribo.iterrows():

        tiempo = int(fila["Tiempo de interarribo (min)"])
        frecuencia = int(fila["Frecuencia"])
        porcentaje = float(fila["Porcentaje"])

        unidad = "minuto" if tiempo == 1 else "minutos"

        st.write(
            f"- En **{frecuencia} ocasiones**, la diferencia de tiempo "
            f"entre un vehículo y el siguiente fue de **{tiempo} {unidad}**, "
            f"lo que representa el **{porcentaje:.0f} %** de las observaciones."
        )
    
    st.info(
        "Los interarribos más frecuentes fueron de 2 y 3 minutos, "
        "con 15 observaciones cada uno, equivalentes al 30 % respectivamente. "
        "El interarribo menos frecuente fue de 5 minutos, con solo 3 observaciones "
        "(6 %)."
    )

    # ========================================================
    # FRECUENCIA RELATIVA
    # ========================================================

    st.markdown("### 📈 Frecuencia relativa")

    st.latex(r"""
    p_i = \frac{f_i}{N}
    """)

    st.markdown(
        r"""
        - $f_i$ es la frecuencia observada;
        - $N=50$ es el número total de observaciones;
        - $p_i$ es la frecuencia relativa.
        """
    )

    st.write("Ejemplo para un interarribo de **1 minuto**:")

    st.latex(r"""
    p(A=1)=\frac{10}{50}=0.20=20\%
    """)

    st.write(
        """
        Esto significa que en el **20 % de las observaciones**
        el tiempo de interarribo fue de 1 minuto.
        """
    )

    # ========================================================
    # MEDIA INTERARRIBOS
    # ========================================================

    st.markdown("### 🧮 Media del tiempo de interarribo")

    st.latex(r"""
    \bar{A}
    =
    \frac{\sum A_i f_i}{\sum f_i}
    """)

    st.latex(r"""
    \bar{A}
    =
    \frac{
    1(10)+2(15)+3(15)+4(7)+5(3)
    }{50}
    """)

    st.latex(r"""
    \bar{A}
    =
    \frac{128}{50}
    =
    2.56\text{ min}
    """)

    st.metric(
        "⏱️ Tiempo medio de interarribo",
        "2.56 min"
    )

    st.success(
        "Interpretación: en promedio, transcurren 2.56 minutos "
        "entre la llegada de un vehículo y el siguiente."
    )

    # ========================================================
    # B. TIEMPOS DE SERVICIO
    # ========================================================

    st.divider()

    st.subheader("⛽ B. Tiempos de servicio")

    st.write(
        """
        El **tiempo de servicio** representa cuánto tarda el surtidor
        en atender completamente a un vehículo.
        """
    )

    datos_servicio = tabla_servicio[
        ["tiempo", "frecuencia", "probabilidad"]
    ].copy()

    datos_servicio.columns = [
        "Tiempo de servicio (min)",
        "Frecuencia",
        "Frecuencia relativa"
    ]

    datos_servicio["Porcentaje"] = (
        datos_servicio["Frecuencia relativa"] * 100
    )

    st.dataframe(
        datos_servicio,
        width="stretch",
        hide_index=True
    )
    
    # ========================================================
    # INTERPRETACIÓN DE LA TABLA
    # ========================================================

    st.markdown("### 🧠 Interpretación de la tabla")

    st.write(
        "La frecuencia indica cuántas observaciones presentaron "
        "cada tiempo de servicio."
    )

    for _, fila in datos_servicio.iterrows():

        tiempo = int(fila["Tiempo de servicio (min)"])
        frecuencia = int(fila["Frecuencia"])
        porcentaje = float(fila["Porcentaje"])

        unidad = "minuto" if tiempo == 1 else "minutos"

        st.write(
            f"- Aparecieron **{frecuencia} observaciones** en las que "
            f"cada vehículo tardó **{tiempo} {unidad}** en ser atendido, "
            f"lo que representa el **{porcentaje:.0f} %** del total."
        )

    st.info(
        "El tiempo de servicio más frecuente fue de 4 minutos: "
        "aparecieron 20 observaciones con ese tiempo, equivalentes al 40 % "
        "del total. Los tiempos menos frecuentes fueron de 2 y 6 minutos, "
        "con 5 observaciones cada uno (10 % respectivamente)."
    )

    # ========================================================
    # FRECUENCIA RELATIVA SERVICIO
    # ========================================================

    st.markdown("### 📈 Frecuencia relativa")

    st.latex(r"""
    p_i = \frac{f_i}{N}
    """)

    st.write("Ejemplo para un servicio de **4 minutos**:")

    st.latex(r"""
    p(S=4)=\frac{20}{50}=0.40=40\%
    """)

    st.write(
        """
        Esto significa que en el **40 % de las observaciones**
        atender un vehículo tardó 4 minutos.
        """
    )

    # ========================================================
    # MEDIA SERVICIO
    # ========================================================

    st.markdown("### 🧮 Media del tiempo de servicio")

    st.latex(r"""
    \bar{S}
    =
    \frac{\sum S_i f_i}{\sum f_i}
    """)

    st.latex(r"""
    \bar{S}
    =
    \frac{
    2(5)+3(10)+4(20)+5(10)+6(5)
    }{50}
    """)

    st.latex(r"""
    \bar{S}
    =
    \frac{200}{50}
    =
    4\text{ min}
    """)

    st.metric(
        "⛽ Tiempo medio de servicio",
        "4.00 min"
    )

    st.success(
        "Interpretación: atender un vehículo requiere, en promedio, 4 minutos."
    )

    # ========================================================
    # COMPARACIÓN
    # ========================================================

    st.divider()

    st.subheader("🧠 Interpretación de los datos observados")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "🚗 Interarribo medio",
            "2.56 min"
        )

    with col2:

        st.metric(
            "⛽ Servicio medio",
            "4.00 min"
        )

    st.write(
        """
        Los datos muestran que un nuevo vehículo aparece, en promedio,
        cada **2.56 minutos**, mientras que atender un vehículo requiere
        aproximadamente **4 minutos**.
        """
    )

    st.latex(r"""
    2.56 < 4
    """)

    st.warning(
        "Esta comparación constituye una primera señal de que pueden "
        "formarse colas, porque los vehículos llegan en promedio más "
        "rápidamente de lo que un único surtidor puede atenderlos."
    )

    # ========================================================
    # RESUMEN PARA DEFENSA
    # ========================================================

    st.divider()

    with st.expander("🎓 ¿Cómo defender esta instrucción en el examen?"):

        st.markdown("**1. ¿Cuántos datos se analizaron?**")

        st.write(
            "Se analizaron 50 observaciones de tiempos de interarribo "
            "y 50 observaciones de tiempos de servicio."
        )


        st.markdown("**2. ¿Cómo se obtuvo la frecuencia relativa?**")

        st.write(
            "Se dividió cada frecuencia observada entre el total de observaciones:"
        )

        st.latex(r"p_i = \frac{f_i}{50}")

        st.write(
            "donde $f_i$ es la frecuencia observada y 50 es el total de observaciones."
        )


        st.markdown("**3. ¿Cuál fue el interarribo medio?**")

        st.latex(r"\bar{A} = 2.56\ \text{min}")


        st.markdown("**4. ¿Cuál fue el servicio medio?**")

        st.latex(r"\bar{S} = 4\ \text{min}")


        st.markdown("**5. ¿Qué interpretación inicial obtenemos?**")

        st.write(
            "Los vehículos llegan, en promedio, con mayor rapidez de la que "
            "el servidor puede atenderlos, lo que constituye una primera "
            "evidencia de posible formación de cola."
        )
