import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import random

# Leer archivo
df_opciones = pd.read_csv("opciones_respuesta.csv")
df_posibles = {
    col: sorted(df_opciones[col].dropna().unique().tolist())
    for col in df_opciones.columns
}

# Agrupar variables
grupo_colegio = ['COLE_BILINGUE', 'COLE_CALENDARIO', 'COLE_CARACTER', 'COLE_GENERO', 'COLE_JORNADA_GRUPO']
grupo_familia = ['FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_TIENEAUTOMOVIL',
                 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA', 'EDU_PADRES_MAX']
grupo_estudiante = ['EDAD', 'ESTU_DEPTO_RESIDE']

# Inicializar app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
    ],
    suppress_callback_exceptions=True
)
app.title = "Predicción Puntaje ICFES"

# Diccionario para renombrar columnas a un lenguaje más natural
nombre_natural = {
    'COLE_BILINGUE': '¿Colegio bilingüe?',
    'COLE_CALENDARIO': 'Calendario académico',
    'COLE_CARACTER': 'Carácter del colegio',
    'COLE_GENERO': 'Género del colegio',
    'COLE_JORNADA_GRUPO': 'Jornada escolar',
    'FAMI_ESTRATOVIVIENDA': 'Estrato de vivienda',
    'FAMI_PERSONASHOGAR': 'Cantidad de personas en el hogar',
    'FAMI_TIENEAUTOMOVIL': '¿Tienen automóvil?',
    'FAMI_TIENECOMPUTADOR': '¿Tienen computador?',
    'FAMI_TIENEINTERNET': '¿Tienen internet?',
    'FAMI_TIENELAVADORA': '¿Tienen lavadora?',
    'EDU_PADRES_MAX': 'Nivel educativo más alto de los padres',
    'EDAD': 'Edad del estudiante',
    'ESTU_DEPTO_RESIDE': 'Departamento de residencia'
}


def crear_grupo(nombre_grupo, icono, variables):
    return html.Div([
        html.H4([html.I(className=f"bi {icono} me-2"), nombre_grupo], className="text-primary mt-4"),
        dbc.Row([
            dbc.Col([
                html.Label(nombre_natural.get(var, var), className="fw-bold"),
                dcc.Dropdown(
                    id=var,
                    options=[{"label": str(val), "value": val} for val in df_posibles[var]],
                    value=df_posibles[var][0],
                    className="mb-3"
                )
            ], width=4) for var in variables
        ])
    ])

layout_prediccion = html.Div([
        # Encabezado con logo
        html.Div(style={
            'backgroundColor': "#5b95bc",
            'padding': '20px 30px',
            'display': 'flex',
            'justifyContent': 'space-between',
            'alignItems': 'center'
        }, children=[
            html.H2("🎓 Predicción del Puntaje Global ICFES", style={'color': 'white', 'margin': 0}),
            html.Img(src='/assets/icfes.png', style={'height': '150px'})
        ]),

        html.Div(
        style={
            'backgroundColor': 'white',
            'borderRadius': '12px',
            'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
            'padding': '20px',
            'margin': '20px 30px',
            'fontSize': '16px',
            'lineHeight': '1.5',
            'color': '#333'
        },
        children=[
            html.Div("🧠 Sobre este dashboard", style={
                'fontWeight': 'bold',
                'fontSize': '18px',
                'marginBottom': '10px'
            }),
            html.P(
                "Este dashboard permite predecir el desempeño en las pruebas ICFES a partir de variables socioeconómicas y características del entorno familiar y educativo del estudiante. "
                "La predicción estima el desempeño en categorías como alto, medio o bajo, con base en la información seleccionada por el usuario."
            )
        ]),

        dbc.Container([

            crear_grupo("Información del Colegio", "bi-building", grupo_colegio),
            crear_grupo("Información Familiar", "bi-people", grupo_familia),
            crear_grupo("Información del Estudiante", "bi-person", grupo_estudiante),

            html.Div(className="text-center my-4", children=[
                html.Button("🔍 Predecir Puntaje", id="btn-pred", n_clicks=0, className="btn btn-primary btn-lg"),
                html.Div(id="output-pred", className="mt-4")
            ])
        ], fluid=True),
    ]),

layout_shap = html.Div([
    html.H3("🔍 Variables más influyentes (SHAP)", style={"marginTop": "20px"}),
    html.P("Este gráfico muestra la importancia de cada variable en la predicción del puntaje global."),
    html.Img(src="/assets/shap_summary.png", style={"width": "100%", "maxWidth": "800px"}),
    ], style={"padding": "30px"}),   


# Layout general
app.layout = html.Div(style={'backgroundColor': '#e6f0fa', 'minHeight': '100vh', 'paddingBottom': '50px'}, children=[
    
    dcc.Location(id="url"),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Predicción", href="/")),
            dbc.NavItem(dbc.NavLink("Explicación SHAP", href="/shap")),
        ],
        brand="Dashboard ICFES",
        color="primary",
        dark=True,
    ),
    html.Div(id="page-content"), 
])


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def mostrar_pagina(pathname):
    if pathname == "/shap":
        return layout_shap
    else:
        # Aquí pones el layout actual de tu página de predicción
        return layout_prediccion


@app.callback(
    Output('output-pred', 'children'),
    Input('btn-pred', 'n_clicks'),
    [State(var, 'value') for var in df_posibles]
)


def predecir_puntaje(n_clicks, *vals):
    if n_clicks == 0:
        return ""

    # Selección aleatoria de categoría
    categoria = random.choice(["Bajo", "Medio", "Alto"])

    # Definimos mensaje y color según la categoría
    if categoria == "Alto":
        color = "success"
    elif categoria == "Medio":
        color = "warning"
    else:  # Bajo
        color = "danger"

    return dbc.Card([
        dbc.CardBody([
            html.H4("📈 Desempeño Estimado", className="card-title"),
            html.H2(categoria, className=f"text-{color}", style={'fontSize': '48px'}),
            html.P(className="card-text")
        ])
    ], className="mx-auto", style={"maxWidth": "400px"})


if __name__ == '__main__':
    app.run_server(debug=True)
