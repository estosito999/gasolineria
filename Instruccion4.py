
import pandas as pd
import streamlit as st


# ============================================================
# RÉPLICAS
# ============================================================

def mostrar_instruccion4(simulacion):

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
