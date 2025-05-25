import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import numpy as np

# Cargar desde archivo externo (modifica el nombre si es .csv o .xlsx)
df_opciones = pd.read_csv("opciones_respuesta.csv")

# Convertir a diccionario agrupado
df_posibles = {
    var: sorted(grupo['opcion'].dropna().unique())
    for var, grupo in df_opciones.groupby('variable')
}

app = dash.Dash(__name__)
app.title = "Predicción Puntaje ICFES"

app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'backgroundColor': '#f9f9f9', 'padding': '30px'}, children=[
    html.H1("🎓 Predicción del Puntaje Global ICFES", style={
        'textAlign': 'center',
        'color': '#2c3e50',
        'marginBottom': '40px'
    }),

    html.Div([
        html.Div([
            html.Label(var, style={'fontWeight': 'bold', 'fontSize': '14px'}),
            dcc.Dropdown(
                options=[{'label': str(v), 'value': v} for v in valores],
                id=var,
                value=valores[0],
                style={'width': '100%'}
            )
        ], style={'width': '300px', 'margin': '10px'}) for var, valores in df_posibles.items()
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center'}),

    html.Div(style={'textAlign': 'center', 'marginTop': '30px'}, children=[
        html.Button("🔍 Predecir Puntaje", id='btn-pred', n_clicks=0,
                    style={
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'padding': '10px 25px',
                        'fontSize': '16px',
                        'border': 'none',
                        'borderRadius': '6px',
                        'cursor': 'pointer'
                    })
    ]),

    html.Div(id='output-pred', style={
        'marginTop': '40px',
        'textAlign': 'center',
        'fontSize': '28px',
        'color': '#27ae60',
        'fontWeight': 'bold'
    })
])

@app.callback(
    Output('output-pred', 'children'),
    Input('btn-pred', 'n_clicks'),
    [State(var, 'value') for var in df_posibles]
)
def predecir_puntaje(n_clicks, *vals):
    if n_clicks == 0:
        return ""

    # Simulación de modelo
    puntaje_simulado = np.random.randint(150, 400)
    return f"📈 Puntaje Global Estimado: {puntaje_simulado}"

if __name__ == '__main__':
    app.run_server(debug=True)
