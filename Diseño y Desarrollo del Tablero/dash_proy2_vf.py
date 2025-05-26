import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

# Leer archivo de entradas
df_entradas = pd.read_csv("entradas.csv", header=None, names=["col_binaria"]).query("not col_binaria.str.startswith('puntaje_cat')", engine='python')
df_entradas = df_entradas.dropna().drop_duplicates()
df_entradas["categoria"] = df_entradas["col_binaria"].apply(lambda x: "_".join(x.split("_")[:-1]))
df_entradas["opcion"] = df_entradas["col_binaria"].apply(lambda x: x.split("_")[-1])

# Diccionario: categoria -> lista de opciones
categorias_opciones = df_entradas.groupby("categoria")["opcion"].unique().apply(sorted).to_dict()
columnas_modelo = df_entradas["col_binaria"].tolist()

# Cargar modelo
modelo = load_model("modelo_binario.h5")

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

def crear_campos():
    nombre_natural = {
        'COLE_BILINGUE': '¿Colegio bilingüe?',
        'COLE_CALENDARIO': 'Calendario académico',
        'COLE_CARACTER': 'Carácter del colegio',
        'COLE_GENERO': 'Género del colegio',
        'COLE_JORNADA': 'Jornada escolar',
        'ESTU_DEPTO_RESIDE': 'Departamento de residencia',
        'ESTU_GENERO': 'Género del estudiante',
        'FAMI_CUARTOSHOGAR': 'Cantidad de cuartos en el hogar',
        'FAMI_EDUCACIONMADRE': 'Educación de la madre',
        'FAMI_EDUCACIONPADRE': 'Educación del padre',
        'FAMI_ESTRATOVIVIENDA': 'Estrato de vivienda',
        'FAMI_PERSONASHOGAR': 'Número de personas en el hogar',
        'FAMI_TIENEAUTOMOVIL': '¿Tienen automóvil?',
        'FAMI_TIENECOMPUTADOR': '¿Tienen computador?',
        'FAMI_TIENEINTERNET': '¿Tienen internet?',
        'FAMI_TIENELAVADORA': '¿Tienen lavadora?',
        'cuartil_edad': 'Edad (años)'
    }

    grupo_colegio = ['COLE_BILINGUE', 'COLE_CALENDARIO', 'COLE_CARACTER', 'COLE_GENERO', 'COLE_JORNADA']
    grupo_familia = ['FAMI_ESTRATOVIVIENDA', 'FAMI_PERSONASHOGAR', 'FAMI_CUARTOSHOGAR',
                     'FAMI_TIENEAUTOMOVIL', 'FAMI_TIENECOMPUTADOR', 'FAMI_TIENEINTERNET', 'FAMI_TIENELAVADORA',
                     'FAMI_EDUCACIONPADRE', 'FAMI_EDUCACIONMADRE']
    grupo_estudiante = ['ESTU_GENERO', 'ESTU_DEPTO_RESIDE', 'cuartil_edad']

    def crear_grupo(nombre, icono, variables):
        return html.Div([
            html.H4([html.I(className=f"bi {icono} me-2"), nombre], className="text-primary mt-4"),
            dbc.Row([
                dbc.Col([
                    html.Label(nombre_natural.get(var, var), className="fw-bold"),
                    dcc.Dropdown(
                        id=var,
                        options=[{"label": "Sí" if val == "S" else "No" if val == "N" else "16 o menos" if val == "Q1" else "17" if val == "Q2" else "18" if val == "Q3" else "19 o más" if val == "Q4" else val, "value": val} for val in categorias_opciones[var]],
                        value=categorias_opciones[var][0],
                        className="mb-3"
                    )
                ], width=4) for var in variables if var in categorias_opciones
            ])
        ])

    return html.Div([
        crear_grupo("Información del Colegio", "bi-building", grupo_colegio),
        crear_grupo("Información Familiar", "bi-people", grupo_familia),
        crear_grupo("Información del Estudiante", "bi-person", grupo_estudiante)
    ])
layout_prediccion = html.Div([
    html.Div(style={'padding': '30px'}, children=[
        html.H2("🎓 Predicción desempeño en las pruebas ICFES", className="mb-4"),
        crear_campos(),
        html.Div(className="text-center my-4", children=[
            html.Button("🔍 Predecir Rendimiento", id="btn-pred", n_clicks=0, className="btn btn-primary btn-lg"),
            html.Div(id="output-pred", className="mt-4")
        ])
    ])
])

layout_shap = html.Div([
    html.H3("🔍 Variables más influyentes (SHAP)", style={"marginTop": "20px"}),
    html.P("Este gráfico muestra la importancia de cada variable en la predicción del puntaje global."),
    html.Img(src="/assets/shap_summary.png", style={"width": "100%", "maxWidth": "800px"}),
], style={"padding": "30px"})

app.layout = html.Div(style={'backgroundColor': '#e6f0fa', 'minHeight': '100vh'}, children=[
    dcc.Location(id="url"),
    dbc.NavbarSimple(
        children=[
            dbc.NavItem(dbc.NavLink("Predicción", href="/")),
            dbc.NavItem(dbc.NavLink("Influencia", href="/shap")),
        ],
        brand="Dashboard ICFES",
        color="primary",
        dark=True,
    ),
    html.Div(id="page-content")
])

def mostrar_pagina(pathname):
    if pathname == "/shap":
        return layout_shap
    else:
        return layout_prediccion

@app.callback(
    Output('output-pred', 'children'),
    Input('btn-pred', 'n_clicks'),
    [State(var, 'value') for var in categorias_opciones]
)
def predecir_puntaje(n_clicks, *vals):
    if n_clicks == 0:
        return ""

    respuestas = dict(zip(categorias_opciones.keys(), vals))
    entrada = {col: 0 for col in columnas_modelo}
    for cat, val in respuestas.items():
        col = f"{cat}_{val}"
        if col in entrada:
            entrada[col] = 1

    entrada_filtrada = {col: entrada[col] for col in columnas_modelo if col in entrada}
    if len(entrada_filtrada) != 100:
        return dbc.Alert(f"❌ Número de columnas incorrecto: se esperaba 100 y se obtuvieron {len(entrada_filtrada)}.", color="danger")

    inputs_lista = [np.array([[entrada_filtrada[col]]], dtype=np.float32) for col in columnas_modelo if col in entrada_filtrada]
    pred = modelo.predict(inputs_lista)[0][0]

    if pred >= 0.5:
        categoria = "Alto"
        color = "success"
    else:
        categoria = "Bajo"
        color = "danger"

    return dbc.Card([
        dbc.CardBody([
            html.H4("📈 Desempeño Estimado", className="card-title"),
            html.H2(categoria, className=f"text-{color}", style={'fontSize': '48px'}),
            html.P(f"(Probabilidad: {pred:.2%})", className="card-text")
        ])
    ], className="mx-auto", style={"maxWidth": "400px"})

if __name__ == '__main__':

    @app.callback(
        Output("page-content", "children"),
        Input("url", "pathname")
    )
    def actualizar_pagina(pathname):
        return mostrar_pagina(pathname)

app.run_server(debug=True)
