import numpy as np

import plotly.graph_objects as go

from plotly.subplots import make_subplots


# ============================================================
# 1. ESCENA INTERACTIVA DE LA GASOLINERA
# ============================================================

def grafico_estado_gasolinera(estado, tiempo, max_cola_visible=8):

    fig = go.Figure()


    # --------------------------------------------------------
    # CARRETERA
    # --------------------------------------------------------

    fig.add_shape(
        type="rect",
        x0=0,
        x1=10,
        y0=0.3,
        y1=2.1,
        fillcolor="#424242",
        line=dict(width=0)
    )


    # --------------------------------------------------------
    # LÍNEA DE CARRETERA
    # --------------------------------------------------------

    for x in np.arange(0, 10, 1.2):

        fig.add_shape(
            type="line",
            x0=x,
            x1=x + 0.6,
            y0=1.2,
            y1=1.2,
            line=dict(
                color="white",
                width=2
            )
        )


    # --------------------------------------------------------
    # SURTIDOR
    # --------------------------------------------------------

    fig.add_shape(
        type="rect",
        x0=7.0,
        x1=8.5,
        y0=2.5,
        y1=4.2,
        fillcolor="#2E7D32",
        line=dict(
            color="#1B5E20",
            width=3
        )
    )


    fig.add_annotation(
        x=7.75,
        y=3.35,
        text="<b>SURTIDOR</b>",
        showarrow=False,
        font=dict(
            color="white",
            size=15
        )
    )


    # --------------------------------------------------------
    # VEHÍCULO EN SERVICIO
    # --------------------------------------------------------

    if estado["atendiendo"]:

        vehiculo = estado["atendiendo"][0]

        fig.add_shape(
            type="rect",
            x0=7.05,
            x1=8.40,
            y0=0.65,
            y1=1.75,
            fillcolor="#1976D2",
            line=dict(
                color="white",
                width=2
            )
        )


        fig.add_annotation(
            x=7.72,
            y=1.20,
            text=f"<b>V{vehiculo}</b>",
            showarrow=False,
            font=dict(
                color="white",
                size=14
            )
        )

    else:

        fig.add_annotation(
            x=7.75,
            y=1.20,
            text="<b>LIBRE</b>",
            showarrow=False,
            font=dict(
                color="#2E7D32",
                size=16
            )
        )


    # --------------------------------------------------------
    # VEHÍCULOS EN COLA
    # --------------------------------------------------------

    cola = estado["esperando"]

    visibles = cola[:max_cola_visible]


    posicion_x = 5.8


    for vehiculo in visibles:

        fig.add_shape(
            type="rect",
            x0=posicion_x - 0.55,
            x1=posicion_x + 0.55,
            y0=0.65,
            y1=1.75,
            fillcolor="#F9A825",
            line=dict(
                color="white",
                width=2
            )
        )


        fig.add_annotation(
            x=posicion_x,
            y=1.20,
            text=f"<b>V{vehiculo}</b>",
            showarrow=False,
            font=dict(
                color="black",
                size=12
            )
        )


        posicion_x -= 1.25


    # --------------------------------------------------------
    # SI LA COLA ES MUY GRANDE
    # --------------------------------------------------------

    if len(cola) > max_cola_visible:

        restantes = (
            len(cola)
            -
            max_cola_visible
        )


        fig.add_annotation(
            x=0.45,
            y=1.2,
            text=f"<b>+{restantes}</b>",
            showarrow=False,
            font=dict(
                color="white",
                size=15
            )
        )


    # --------------------------------------------------------
    # INFORMACIÓN
    # --------------------------------------------------------

    fig.add_annotation(
        x=0.3,
        y=4.4,
        xanchor="left",
        text=(
            f"<b>Tiempo:</b> {tiempo} min"
            f"<br><b>Esperando:</b> {len(estado['esperando'])}"
            f"<br><b>Finalizados:</b> {len(estado['finalizados'])}"
        ),
        showarrow=False,
        align="left",
        font=dict(
            size=13
        )
    )


    fig.update_layout(

        title="Estado actual de la gasolinera",

        height=450,

        xaxis=dict(
            range=[0, 10],
            visible=False
        ),

        yaxis=dict(
            range=[0, 5],
            visible=False
        ),

        margin=dict(
            l=10,
            r=10,
            t=55,
            b=10
        ),

        plot_bgcolor="#E8F5E9",

        showlegend=False
    )


    return fig


# ============================================================
# 2. EVOLUCIÓN DE LA COLA EN EL TIEMPO
# ============================================================

def grafico_cola_tiempo(df):

    eventos = sorted(
        set(
            [0]
            +
            df["llegada"].tolist()
            +
            df["inicio_servicio"].tolist()
            +
            df["salida"].tolist()
        )
    )


    tiempos = []

    colas = []


    for t in eventos:

        cola = (
            (
                (df["llegada"] <= t)
                &
                (df["inicio_servicio"] > t)
            )
            .sum()
        )


        tiempos.append(t)

        colas.append(int(cola))


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=tiempos,

            y=colas,

            mode="lines",

            line_shape="hv",

            fill="tozeroy",

            name="Vehículos esperando",

            hovertemplate=(
                "Tiempo: %{x} min"
                "<br>Cola: %{y} vehículos"
                "<extra></extra>"
            )
        )
    )


    fig.update_layout(

        title="Evolución de la longitud de la cola",

        xaxis_title="Tiempo simulado (min)",

        yaxis_title="Vehículos esperando",

        hovermode="x unified",

        height=430
    )


    return fig


# ============================================================
# 3. TIEMPO DE ESPERA DE CADA VEHÍCULO
# ============================================================

def grafico_esperas(df):

    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=df["vehiculo"],

            y=df["espera_cola"],

            mode="lines+markers",

            name="Espera",

            hovertemplate=(
                "Vehículo: V%{x}"
                "<br>Espera: %{y} min"
                "<extra></extra>"
            )
        )
    )


    media = float(
        df["espera_cola"].mean()
    )


    fig.add_hline(

        y=media,

        line_dash="dash",

        annotation_text=f"Media = {media:.2f} min",

        annotation_position="top left"
    )


    fig.update_layout(

        title="Tiempo de espera por vehículo",

        xaxis_title="Vehículo",

        yaxis_title="Tiempo de espera (min)",

        height=430
    )


    return fig


# ============================================================
# 4. LLEGADAS VS SALIDAS ACUMULADAS
# ============================================================

def grafico_flujo_acumulado(df):

    eventos = sorted(
        set(
            [0]
            +
            df["llegada"].tolist()
            +
            df["salida"].tolist()
        )
    )


    llegadas_acumuladas = []

    salidas_acumuladas = []


    for t in eventos:

        llegadas = (
            df["llegada"]
            <= t
        ).sum()


        salidas = (
            df["salida"]
            <= t
        ).sum()


        llegadas_acumuladas.append(
            int(llegadas)
        )


        salidas_acumuladas.append(
            int(salidas)
        )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=eventos,

            y=llegadas_acumuladas,

            mode="lines",

            line_shape="hv",

            name="Llegadas acumuladas"
        )
    )


    fig.add_trace(

        go.Scatter(

            x=eventos,

            y=salidas_acumuladas,

            mode="lines",

            line_shape="hv",

            name="Salidas acumuladas"
        )
    )


    fig.update_layout(

        title="Llegadas y salidas acumuladas",

        xaxis_title="Tiempo simulado (min)",

        yaxis_title="Número de vehículos",

        hovermode="x unified",

        height=430
    )


    return fig


# ============================================================
# 5. LÍNEA TEMPORAL DE VEHÍCULOS
# ============================================================

def grafico_timeline(df, max_vehiculos=20):

    datos = (
        df
        .head(max_vehiculos)
        .copy()
    )


    etiquetas = [

        f"V{int(v)}"

        for v in datos["vehiculo"]
    ]


    fig = go.Figure()


    # --------------------------------------------------------
    # ESPERA
    # --------------------------------------------------------

    fig.add_trace(

        go.Bar(

            y=etiquetas,

            x=datos["espera_cola"],

            base=datos["llegada"],

            orientation="h",

            name="Esperando",

            customdata=datos[
                [
                    "llegada",
                    "inicio_servicio"
                ]
            ],

            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Llegada: %{customdata[0]}"
                "<br>Inicio: %{customdata[1]}"
                "<br>Espera: %{x} min"
                "<extra></extra>"
            )
        )
    )


    # --------------------------------------------------------
    # SERVICIO
    # --------------------------------------------------------

    fig.add_trace(

        go.Bar(

            y=etiquetas,

            x=datos["servicio"],

            base=datos["inicio_servicio"],

            orientation="h",

            name="En servicio",

            customdata=datos[
                [
                    "inicio_servicio",
                    "salida"
                ]
            ],

            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Inicio: %{customdata[0]}"
                "<br>Salida: %{customdata[1]}"
                "<br>Servicio: %{x} min"
                "<extra></extra>"
            )
        )
    )


    fig.update_layout(

        title=(
            f"Línea temporal — primeros "
            f"{len(datos)} vehículos"
        ),

        xaxis_title="Tiempo simulado (min)",

        yaxis_title="Vehículo",

        barmode="overlay",

        height=max(
            450,
            len(datos) * 26
        ),

        hovermode="closest"
    )


    fig.update_yaxes(
        autorange="reversed"
    )


    return fig


# ============================================================
# 6. DISTRIBUCIÓN EMPÍRICA
# ============================================================

def grafico_distribucion(tabla, titulo):

    fig = make_subplots(

        specs=[
            [
                {
                    "secondary_y": True
                }
            ]
        ]
    )


    # --------------------------------------------------------
    # PROBABILIDAD
    # --------------------------------------------------------

    fig.add_trace(

        go.Bar(

            x=tabla["tiempo"],

            y=tabla["probabilidad"],

            name="Probabilidad",

            hovertemplate=(
                "Tiempo: %{x} min"
                "<br>P = %{y:.3f}"
                "<extra></extra>"
            )
        ),

        secondary_y=False
    )


    # --------------------------------------------------------
    # ACUMULADA
    # --------------------------------------------------------

    fig.add_trace(

        go.Scatter(

            x=tabla["tiempo"],

            y=tabla["acumulada"],

            mode="lines+markers",

            name="Acumulada F(x)",

            hovertemplate=(
                "Tiempo: %{x} min"
                "<br>F(x) = %{y:.3f}"
                "<extra></extra>"
            )
        ),

        secondary_y=True
    )


    fig.update_layout(

        title=titulo,

        xaxis_title="Tiempo (min)",

        height=430
    )


    fig.update_yaxes(

        title_text="Probabilidad",

        secondary_y=False
    )


    fig.update_yaxes(

        title_text="Probabilidad acumulada",

        range=[0, 1.05],

        secondary_y=True
    )


    return fig


# ============================================================
# 7. COMPARACIÓN ENTRE RÉPLICAS
# ============================================================

def grafico_replicas(resultados):

    media_global = float(
        resultados["espera_media"].mean()
    )


    fig = go.Figure()


    fig.add_trace(

        go.Scatter(

            x=resultados["replica"],

            y=resultados["espera_media"],

            mode="lines+markers",

            name="Espera media",

            customdata=resultados[
                [
                    "semilla",
                    "cola_maxima",
                    "espera_maxima"
                ]
            ],

            hovertemplate=(
                "<b>Réplica %{x}</b>"
                "<br>Espera media: %{y:.2f} min"
                "<br>Semilla: %{customdata[0]}"
                "<br>Cola máxima: %{customdata[1]}"
                "<br>Espera máxima: %{customdata[2]:.2f}"
                "<extra></extra>"
            )
        )
    )


    fig.add_hline(

        y=media_global,

        line_dash="dash",

        annotation_text=(
            f"Media global = "
            f"{media_global:.2f} min"
        ),

        annotation_position="top left"
    )


    fig.update_layout(

        title="Comparación de las réplicas",

        xaxis_title="Réplica",

        yaxis_title="Espera media (min)",

        height=430
    )


    return fig