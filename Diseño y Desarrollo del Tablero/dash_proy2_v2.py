import dash
from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Leer archivos
df_entradas = pd.read_csv("entradas.csv")
df_encoded = pd.read_csv("df_encoded.csv")
modelo = load_model("modelo_binario.h5")

# Diccionario de opciones por variable
df_posibles = {
    col: sorted(df_entradas[col].dropna().unique().tolist())
    for col in df_entradas.columns
}

# Columnas usadas por el modelo (excluyendo variable objetivo)
columnas_modelo = df_encoded.drop(columns=['puntaje_global']).columns

# Lista completa de variables de entrada
todas_las_entradas = list(df_posibles.keys())

# Inicializar app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"
    ],
    suppress_callback_exceptions=True
)
app.title = "Predicción desempeño en las pruebas ICFES"

# Nombres más legibles
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

def crear_campos():
    return dbc.Row([
        dbc.Col([
            html.Label(nombre_natural.get(var, var), className="fw-bold"),
            dcc.Dropdown(
                id=var,
                options=[{"label": str(val), "value": val} for val in df_posibles[var]],
                value=df_posibles[var][0],
                className="mb-3"
            )
        ], width=4) for var in todas_las_entradas
    ])

layout_prediccion = html.Div([
    html.Div(style={
        'backgroundColor': "#5b95bc",
        'padding': '20px 30px',
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center'
    }, children=[
        html.H2("\ud83c\udf93 Predicci\u00f3n desempe\u00f1o en las pruebas ICFES", style={'color': 'white', 'margin': 0}),
        html.Img(src='/assets/icfes.png', style={'height': '150px'})
    ]),

    html.Div(style={
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'boxShadow': '0 4px 8px rgba(0, 0, 0, 0.1)',
        'padding': '20px',
        'margin': '20px 30px',
        'fontSize': '16px',
        'lineHeight': '1.5',
        'color': '#333'
    }, children=[
        html.Div("\ud83e\udde0 Sobre este dashboard", style={
            'fontWeight': 'bold',
            'fontSize': '18px',
            'marginBottom': '10px'
        }),
        html.P("Este dashboard permite predecir el desempe\u00f1o en las pruebas ICFES a partir de variables socioecon\u00f3micas y caracter\u00edsticas del entorno familiar y educativo del estudiante. La predicci\u00f3n estima el desempe\u00f1o en categor\u00edas como alto o bajo, con base en la informaci\u00f3n seleccionada por el usuario. Por favor, complete todos los campos requeridos para obtener una predicci\u00f3n sobre el rendimiento del estudiante. Adem\u00e1s, en la pesta\u00f1a Influencia se incluye un an\u00e1lisis de las variables m\u00e1s influyentes en la predicci\u00f3n.")
    ]),

    dbc.Container([
        crear_campos(),
        html.Div(className="text-center my-4", children=[
            html.Button("\ud83d\udd0d Predecir Rendimiento", id="btn-pred", n_clicks=0, className="btn btn-primary btn-lg"),
            html.Div(id="output-pred", className="mt-4")
        ])
    ], fluid=True),
])

layout_shap = html.Div([
    html.H3("\ud83d\udd0d Variables m\u00e1s influyentes (SHAP)", style={"marginTop": "20px"}),
    html.P("Este gr\u00e1fico muestra la importancia de cada variable en la predicci\u00f3n del puntaje global."),
    html.Img(src="/assets/shap_summary.png", style={"width": "100%", "maxWidth": "800px"}),
], style={"padding": "30px"})

app.layout = html.Div(style={'backgroundColor': '#e6f0fa', 'minHeight': '100vh', 'paddingBottom': '50px'}, children=[
    dcc.Location(id="url"),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Predicci\u00f3n", href="/")),
            dbc.NavItem(dbc.NavLink("Influencia", href="/shap")),
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
        return layout_prediccion

@app.callback(
    Output('output-pred', 'children'),
    Input('btn-pred', 'n_clicks'),
    [State(var, 'value') for var in todas_las_entradas]
)
def predecir_puntaje(n_clicks, *vals):
    if n_clicks == 0:
        return ""

    entrada_dict = dict(zip(todas_las_entradas, vals))
    entrada_df = pd.DataFrame([entrada_dict])

    entrada_encoded = pd.get_dummies(entrada_df).reindex(columns=columnas_modelo, fill_value=0)

    try:
        pred = modelo.predict(entrada_encoded)[0][0]
    except Exception as e:
        return dbc.Alert(f"Error en la predicci\u00f3n: {str(e)}", color="danger")

    categoria = "Alto" if pred >= 0.5 else "Bajo"
    color = "success" if categoria == "Alto" else "danger"

    return dbc.Card([
        dbc.CardBody([
            html.H4("\ud83d\udcc8 Desempe\u00f1o Estimado", className="card-title"),
            html.H2(categoria, className=f"text-{color}", style={'fontSize': '48px'}),
            html.P(f"Probabilidad: {pred:.2%}", className="card-text")
        ])
    ], className="mx-auto", style={"maxWidth": "400px"})

if __name__ == '__main__':
    app.run_server(debug=True)
