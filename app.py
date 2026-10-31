import math

import streamlit as st
from scipy.stats import t as student_t


from datos import LLEGADAS
from datos import SERVICIO
from datos import obtener_distribuciones
from datos import resumen_datos

from generador import *

from simulador import *

from interarribos import *

from experimentos import ejecutar_replicas

from estadistica import *

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

datos_base = resumen_datos()



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


# ============================================================
# DATOS
# ============================================================
with tab1:

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

# ============================================================
# TRANSFORMADA INVERSA
# ============================================================

with tab2:

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
# ============================================================
# ANÁLISIS DE DISTRIBUCIÓN Y SIMULACIÓN EMPÍRICA
# ============================================================

with tab3:

    st.header("📐 Instrucción 3 — Análisis de distribución")

    st.info(
        "Realizar el análisis de distribución. Plantear una hipótesis de distribución, "
        "calcular las frecuencias esperadas y aplicar chi-cuadrado cuando se cumplan sus "
        "condiciones. Si no se ajusta una distribución teórica, pueden utilizar una "
        "distribución empírica discreta para simular, justificándolo."
    )
    st.write(
        "Se tratan los tiempos registrados como valores discretos exactos en minutos, "
        "no como intervalos de clase. Poisson es una hipótesis sobre esos valores "
        "enteros; este ajuste no demuestra que las llegadas sigan un proceso de Poisson. "
        "La independencia y estabilidad de las observaciones se suponen para la prueba; "
        "no pueden verificarse con estas frecuencias resumidas."
    )

    alpha_ajuste = 0.05
    st.subheader("Hipótesis y nivel de significancia")
    st.markdown(
        "**H0:** Los datos son compatibles con la distribución Poisson propuesta.\n\n"
        "**H1:** Los datos no son compatibles con dicha distribución."
    )
    st.latex(r"\alpha = 0.05")

    st.write(
        "Para cada tabla se estima λ con la media ponderada de los mismos datos. "
        "Las cinco filas de cada tabla resumen 50 observaciones; las filas no son "
        "cinco observaciones individuales."
    )
    st.latex(r"N=\sum_x f_x,\qquad \widehat{\lambda}=\frac{\sum_x x f_x}{N}")
    st.caption(
        "λ estimado es el parámetro de Poisson para los valores registrados. "
        "Es distinto de la tasa de llegadas 1 / interarribo medio de la Instrucción 1."
    )

    st.subheader("Frecuencias esperadas y condiciones de la prueba")
    st.latex(
        r"E_i=N\,P_{\widehat{\lambda}}(X\in C_i),\qquad "
        r"\chi^2=\sum_{i=1}^{k}\frac{(O_i-E_i)^2}{E_i},\qquad gl=k-1-1=k-2"
    )
    st.write(
        "Se incluyen las categorías desde cero, aun si no se observaron, y la cola "
        "por encima del máximo observado. Se fusionan categorías contiguas según "
        "sus frecuencias esperadas hasta obtener Eᵢ ≥ 5; el sobrante final se une "
        "a la clase anterior. Las esperadas se suman al agrupar, sin redondearlas. "
        "k es el número de clases finales: se descuenta un grado por el total fijo "
        "y otro porque λ se estimó con la misma muestra. Si no quedan grados de "
        "libertad positivos o no se alcanza el mínimo esperado, no se aplica la prueba. "
        "El p-valor corresponde a una aproximación asintótica."
    )

    for nombre_datos, frecuencias in (
        ("A. Tiempos de interarribo", LLEGADAS),
        ("B. Tiempos de servicio", SERVICIO),
    ):
        st.divider()
        st.subheader(nombre_datos)
        ajuste = ajuste_poisson(frecuencias, alpha=alpha_ajuste)
        tabla_ajuste = ajuste["tabla"]

        st.metric("λ estimado con los mismos datos", f"{ajuste['lambda_estimado']:.4f}")
        st.caption(
            f"N = {ajuste['n']} observaciones; "
            f"{ajuste['clases_originales']} categorías, incluida la cola, "
            f"se agrupan en k = {len(tabla_ajuste)} clases. "
            "El signo + indica categorías fusionadas y >= indica la cola."
        )

        st.dataframe(
            tabla_ajuste[["clase", "observada", "esperada", "contribucion"]].rename(
                columns={
                    "clase": "Clase",
                    "observada": "Frecuencia observada O_i",
                    "esperada": "Frecuencia esperada E_i",
                    "contribucion": "(O_i - E_i)^2 / E_i",
                }
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Σ Oᵢ = {tabla_ajuste['observada'].sum():.0f}; "
            f"Σ Eᵢ = {tabla_ajuste['esperada'].sum():.4f}; "
            f"mínimo Eᵢ después de agrupar = {tabla_ajuste['esperada'].min():.4f}."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "χ² calculado",
            f"{ajuste['chi2']:.6f}" if ajuste["valido"] else "No aplicable",
        )
        c2.metric("Grados de libertad", ajuste["gl"])
        c3.metric(
            "p-valor", f"{ajuste['pvalor']:.6g}" if ajuste["valido"] else "No aplicable"
        )
        c4.metric("α", f"{ajuste['alpha']:.2f}")

        st.markdown("**Decisión estadística**")
        if not ajuste["valido"]:
            st.warning(
                f"No se aplica chi-cuadrado: {ajuste['motivo']} "
                "No se emite un p-valor ni una decisión de rechazo."
            )
        elif ajuste["rechazar"]:
            st.warning(
                f"p = {ajuste['pvalor']:.6g} < α = {ajuste['alpha']:.2f} → rechazar H0. "
                "Existe evidencia estadística para rechazar la compatibilidad "
                "con la distribución Poisson propuesta al nivel de significancia utilizado."
            )
        else:
            st.info(
                f"p = {ajuste['pvalor']:.6g} ≥ α = {ajuste['alpha']:.2f} → no rechazar H0. "
                "No existe evidencia estadística suficiente para rechazar H0 al nivel "
                "de significancia utilizado. Esto no demuestra que la distribución sea Poisson."
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


# ============================================================
# RÉPLICAS
# ============================================================

with tab4:

    # ========================================================
    # INSTRUCCIÓN 4
    # ========================================================

    st.header("⛽ Instrucción 4 — Construcción del modelo")

    st.info(
        "Construir el modelo. Programar o utilizar ProModel, con una entidad cliente, "
        "una cola FIFO y un servidor."
    )


    # ========================================================
    # 1. ELEMENTOS DEL MODELO
    # ========================================================

    st.subheader("🏗️ Elementos del modelo")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("### 🚗 Entidad")

        st.write(
            "La entidad del sistema es el **vehículo**. "
            "Cada vehículo llega a la gasolinera, puede esperar "
            "en cola y posteriormente recibe servicio."
        )

    with col2:

        st.markdown("### 🚦 Cola")

        st.write(
            "Existe una sola cola y utiliza la disciplina **FIFO**."
        )

        st.latex(
            r"\text{FIFO}=\text{First In, First Out}"
        )

        st.write(
            "El primer vehículo en llegar a la cola es el primero "
            "en ser atendido."
        )

    with col3:

        st.markdown("### ⛽ Servidor")

        st.write(
            "El sistema dispone de un único servidor, representado "
            "por un surtidor de combustible."
        )


    # ========================================================
    # 2. FLUJO DEL SISTEMA
    # ========================================================

    st.divider()

    st.subheader("🔄 Flujo del sistema")

    st.code(
        """
Vehículo
    ↓
Llegada a la gasolinera
    ↓
¿El surtidor está libre?
    ↓
 ┌───────────────┐
 │               │
Sí              No
 │               │
 ↓               ↓
Servicio      Cola FIFO
 │               │
 │        Esperar turno
 │               │
 └───────←───────┘
        ↓
     Servicio
        ↓
      Salida
        """,
        language=None
    )

    st.write(
        "Cuando un vehículo llega, puede comenzar inmediatamente "
        "su servicio si el surtidor está libre. Si el surtidor está "
        "ocupado, debe esperar en la cola FIFO."
    )


    # ========================================================
    # 3. VARIABLES DEL MODELO
    # ========================================================

    st.divider()

    st.subheader("📐 Variables utilizadas")

    variables_modelo = pd.DataFrame({
        "Variable": [
            "Aᵢ",
            "Tᵢ",
            "Sᵢ",
            "Bᵢ",
            "Dᵢ",
            "Wqᵢ"
        ],
        "Significado": [
            "Tiempo de interarribo del vehículo i",
            "Momento de llegada del vehículo i",
            "Tiempo de servicio del vehículo i",
            "Momento en que inicia el servicio",
            "Momento en que termina el servicio y sale",
            "Tiempo de espera en cola"
        ],
        "Unidad": [
            "min",
            "min",
            "min",
            "min",
            "min",
            "min"
        ]
    })

    st.dataframe(
        variables_modelo,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 4. MOMENTO DE LLEGADA
    # ========================================================

    st.divider()

    st.subheader("🚗 1. Momento de llegada")

    st.write(
        "El momento de llegada se obtiene acumulando los tiempos "
        "de interarribo."
    )

    st.latex(
        r"T_i=T_{i-1}+A_i"
    )

    st.write(
        "Por ejemplo, si el vehículo anterior llegó en el minuto 6 "
        "y el nuevo interarribo es de 2 minutos:"
    )

    st.latex(
        r"T_i=6+2=8"
    )

    st.write(
        "El nuevo vehículo llegará en el minuto 8."
    )


    # ========================================================
    # 5. INICIO DEL SERVICIO
    # ========================================================

    st.divider()

    st.subheader("⛽ 2. Inicio del servicio")

    st.write(
        "El vehículo solamente puede iniciar su servicio cuando "
        "ya ha llegado y el surtidor se encuentra libre."
    )

    st.latex(
        r"B_i=\max(T_i,D_{i-1})"
    )

    st.write(
        "Por eso se utiliza el máximo entre el momento de llegada "
        "del vehículo y la salida del vehículo anterior."
    )

    st.markdown("**Caso 1 — El surtidor está libre**")

    st.latex(
        r"T_i>D_{i-1}\quad\Rightarrow\quad B_i=T_i"
    )

    st.write(
        "El vehículo comienza a ser atendido inmediatamente."
    )

    st.markdown("**Caso 2 — El surtidor está ocupado**")

    st.latex(
        r"T_i<D_{i-1}\quad\Rightarrow\quad B_i=D_{i-1}"
    )

    st.write(
        "El vehículo debe esperar hasta que termine el servicio anterior."
    )


    # ========================================================
    # 6. FINALIZACIÓN DEL SERVICIO
    # ========================================================

    st.divider()

    st.subheader("🏁 3. Finalización del servicio")

    st.write(
        "Una vez que comienza el servicio, sumamos el tiempo de servicio "
        "para determinar cuándo sale el vehículo."
    )

    st.latex(
        r"D_i=B_i+S_i"
    )

    st.write(
        "Si el servicio comienza en el minuto 8 y tarda 4 minutos:"
    )

    st.latex(
        r"D_i=8+4=12"
    )

    st.write(
        "El vehículo termina de ser atendido y sale en el minuto 12."
    )


    # ========================================================
    # 7. TIEMPO DE ESPERA
    # ========================================================

    st.divider()

    st.subheader("⏳ 4. Tiempo de espera en cola")

    st.write(
        "El tiempo de espera se obtiene restando el momento de llegada "
        "al momento en que comienza el servicio."
    )

    st.latex(
        r"W_{q,i}=B_i-T_i"
    )

    st.write(
        "Si un vehículo llega en el minuto 8 pero comienza su servicio "
        "en el minuto 12:"
    )

    st.latex(
        r"W_{q,i}=12-8=4\text{ min}"
    )

    st.write(
        "Ese vehículo esperó 4 minutos en la cola."
    )


    # ========================================================
    # 8. RESUMEN MATEMÁTICO
    # ========================================================

    st.divider()

    st.subheader("🧠 Lógica matemática completa")

    st.latex(
        r"T_i=T_{i-1}+A_i"
    )

    st.latex(
        r"B_i=\max(T_i,D_{i-1})"
    )

    st.latex(
        r"D_i=B_i+S_i"
    )

    st.latex(
        r"W_{q,i}=B_i-T_i"
    )

    st.code(
        """
Interarribo
     ↓
Llegada
     ↓
Inicio del servicio
     ↓
Finalización
     ↓
Tiempo de espera
        """,
        language=None
    )


    # ========================================================
    # 9. CONDICIONES DEL SISTEMA
    # ========================================================

    st.divider()

    st.subheader("⚙️ Condiciones del modelo")

    condiciones = pd.DataFrame({
        "Elemento": [
            "Sistema",
            "Disciplina",
            "Servidores",
            "Unidad de tiempo",
            "Clientes por réplica",
            "Estado inicial"
        ],
        "Condición": [
            "Una cola y un servidor",
            "FIFO",
            "1 surtidor",
            "Minutos",
            "100 vehículos",
            "Servidor libre y cola vacía"
        ]
    })

    st.dataframe(
        condiciones,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 10. TABLA DE RESULTADOS DEL MODELO
    # ========================================================

    st.divider()

    st.subheader("📋 Registro de vehículos simulados")

    st.write(
        "La siguiente tabla permite verificar el recorrido de cada vehículo "
        "desde su llegada hasta su salida del sistema."
    )

    columnas_modelo = [
        "vehiculo",
        "u_llegada",
        "entre_llegadas",
        "llegada",
        "u_servicio",
        "servicio",
        "inicio_servicio",
        "salida",
        "espera_cola",
        "tiempo_sistema"
    ]

    tabla_modelo = simulacion[columnas_modelo].copy()

    tabla_modelo.columns = [
        "Vehículo",
        "U interarribo",
        "Interarribo (min)",
        "Llegada",
        "U servicio",
        "Servicio (min)",
        "Inicio servicio",
        "Salida",
        "Espera en cola",
        "Tiempo en sistema"
    ]

    st.dataframe(
        tabla_modelo,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 11. TRES CLIENTES CONSECUTIVOS
    # ========================================================

    st.divider()

    st.subheader("🔍 Ejemplo verificable de tres vehículos consecutivos")

    st.write(
        "Para observar claramente la lógica FIFO, tomamos los primeros "
        "tres vehículos de la simulación."
    )

    tres_clientes = tabla_modelo.head(3)

    st.dataframe(
        tres_clientes,
        width="stretch",
        hide_index=True
    )


    # ========================================================
    # 12. RECONSTRUCCIÓN AUTOMÁTICA
    # ========================================================

    for i, fila in tres_clientes.iterrows():

        vehiculo = int(fila["Vehículo"])

        with st.expander(f"🚗 Vehículo {vehiculo}"):

            st.write(
                f"Interarribo generado: **{fila['Interarribo (min)']} min**"
            )

            st.write(
                f"Momento de llegada: **{fila['Llegada']:.2f} min**"
            )

            st.write(
                f"Tiempo de servicio: **{fila['Servicio (min)']} min**"
            )

            st.write(
                f"Inicio del servicio: **{fila['Inicio servicio']:.2f} min**"
            )

            st.write(
                f"Salida del sistema: **{fila['Salida']:.2f} min**"
            )

            st.write(
                f"Espera en cola: **{fila['Espera en cola']:.2f} min**"
            )

            st.write(
                f"Tiempo total en el sistema: "
                f"**{fila['Tiempo en sistema']:.2f} min**"
            )


    # ========================================================
    # 13. INTERPRETACIÓN DEL MODELO
    # ========================================================

    st.divider()

    st.subheader("💡 Interpretación")

    st.write(
        "El modelo reproduce el comportamiento de una gasolinera con "
        "un único surtidor. Los vehículos se generan utilizando los "
        "tiempos obtenidos en la Instrucción 2 y son atendidos respetando "
        "estrictamente el orden FIFO."
    )

    st.success(
        "El modelo contiene la entidad vehículo, una cola FIFO, un único "
        "servidor y registra correctamente las llegadas, esperas, servicios "
        "y salidas."
    )

# ============================================================
# EJECUCIÓN DE RÉPLICAS
# ============================================================

with tab5:

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

with tab6:

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