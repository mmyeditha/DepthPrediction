import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objects as go
import os

# get all of the folders containing CSV mesh data
# each folder contains multiple CSV files (one file per possible mesh classification)
prefixes = os.listdir('3d_files')

#declare the Plotly Dash app
app = dash.Dash(__name__)

#define the layout of the app
app.layout = html.Div([
    html.P("Choose a Saved Mesh:"),
    dcc.Dropdown(
        id='dropdown',
        options=[{'value': x, 'label': x}
                 for x in prefixes],
        value=prefixes[0],
        clearable=False
    ),
    dcc.Graph(id="graph"),
], style={"fontFamily": "sans-serif"})

#link the dropdown to the graph, so when the dropdown item is changed, the graph will update
@app.callback(
    Output("graph", "figure"),
    [Input("dropdown", "value")]
)

def display_mesh(name): #name = the selected value from the dropdown
    fig = go.Figure()
    #add one trace per mesh classification for the given route
    file_path = f"3d_files/{name}/-ply.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        fig.add_trace(go.Scatter3d(
            x=df.x, y=df.y, z=df.z,
            mode="markers",
            #i=df.i.iloc[:500], j=df.j.iloc[:500], k=df.k.iloc[:500],
            marker=dict(
                color="green",
                size=1
            ),
            showlegend=True
        ))

    fig.update_layout(width=900, height=700, scene = dict(
        xaxis = dict(nticks=4, range=[-35,10],),
        yaxis = dict(nticks=4, range=[-35,10],),
        zaxis = dict(nticks=4, range=[-35,10],),),)
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
