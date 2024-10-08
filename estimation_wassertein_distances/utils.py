import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Create a Dash app
app = dash.Dash(__name__)

# Define your custom functions f_x(u) and f_y(u)
def f_x(u):
    return 2*np.cos(u)  # Change this to any function of u for x

def f_y(u):
    return np.sin(u) # Change this to any function of u for y

# Define parameters for the torus
R = 3  # Major radius
r = 1  # Minor radius
u = np.linspace(0, 2 * np.pi, 1000)  # Parametric variable u

# Torus equations using f_x(u) and f_y(u)
v_x = f_x(u)  # Map function to the torus parameter for x
v_y = f_y(u)  # Map function to the torus parameter for y
x_curve = (R + r * np.cos(v_x)) * np.cos(v_y)
y_curve = (R + r * np.cos(v_x)) * np.sin(v_y)
z_curve = r * np.sin(v_x)

# Create the surface for the torus
u_mesh, v_mesh = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, 2 * np.pi, 100))
x_torus = (R + r * np.cos(v_mesh)) * np.cos(u_mesh)
y_torus = (R + r * np.cos(v_mesh)) * np.sin(u_mesh)
z_torus = r * np.sin(v_mesh)

# Create traces for the initial plot
trace_torus_surface = go.Surface(x=x_torus, y=y_torus, z=z_torus, opacity=0.6, colorscale='Blues', showscale=False)
trace_curve_on_torus = go.Scatter3d(x=x_curve, y=y_curve, z=z_curve, mode='lines', line=dict(color='red', width=5), name="Function on Torus")

# Create the plot of (f_x(u), f_y(u)) in R^2
trace_curve_xy = go.Scatter(x=f_x(u), y=f_y(u), mode='lines', line=dict(color='purple', width=3), name="(f_x(u), f_y(u)) in R^2")

# Layout for the initial plots
app.layout = html.Div([
    html.Div([
        dcc.Graph(
            id='torus_graph',
            figure={
                'data': [trace_torus_surface, trace_curve_on_torus],
                'layout': go.Layout(
                    title='Function on Torus',
                    scene=dict(
                        xaxis=dict(title='X', range=[-5, 5]),  # Set appropriate ranges for axes
                        yaxis=dict(title='Y', range=[-5, 5]),
                        zaxis=dict(title='Z', range=[-5, 5]),
                        aspectmode='manual', 
                        aspectratio=dict(x=1, y=1, z=1)  # Equal scaling for axes
                    ),
                    width=800,  # Set the width of the graph
                    height=800,  # Set the height of the graph
                )
            }
        ),
    ], style={'width': '49%', 'display': 'inline-block'}),

    html.Div([
        dcc.Graph(
            id='function_graph',
            figure={
                'data': [trace_curve_xy],
                'layout': go.Layout(
                    title='(f_x(u), f_y(u)) in R^2',
                    xaxis=dict(title='f_x(u)'),
                    yaxis=dict(title='f_y(u)'),
                )
            }
        ),
        html.Div(id='output', style={'marginTop': 20})
    ], style={'width': '49%', 'display': 'inline-block', 'vertical-align': 'top'})
])

# Callback to update the point on the torus when hovering over the 2D function
@app.callback(
    Output('torus_graph', 'figure'),
    [Input('function_graph', 'hoverData')],
    [State('torus_graph', 'relayoutData')]  # Capture the current camera view
)
def update_torus_on_hover(hoverData, relayoutData):
    # Start with the base plot (surface and curve on torus)
    base_figure = go.Figure(data=[trace_torus_surface, trace_curve_on_torus])

    if hoverData:
        # Extract the u value from the hover event
        f_x_hover = hoverData['points'][0]['x']
        f_y_hover = hoverData['points'][0]['y']

        # Find the corresponding u value (use inverse of f_x and f_y to get u if needed)
        u_hover = np.argmin(np.abs(f_x(u) - f_x_hover))  # Approximate u based on f_x

        v_x_hover = f_x(u[u_hover])  # Get the corresponding v_x value using f_x(u)
        v_y_hover = f_y(u[u_hover])  # Get the corresponding v_y value using f_y(u)

        # Calculate the corresponding 3D point on the torus
        x_hover = (R + r * np.cos(v_x_hover)) * np.cos(v_y_hover)
        y_hover = (R + r * np.cos(v_x_hover)) * np.sin(v_y_hover)
        z_hover = r * np.sin(v_x_hover)

        # Add the point on the torus
        hover_point = go.Scatter3d(x=[x_hover], y=[y_hover], z=[z_hover], mode='markers', marker=dict(color='black', size=8), name="Hover Point")
        base_figure.add_trace(hover_point)

    # Preserve the camera view if it exists in relayoutData
    if relayoutData and 'scene.camera' in relayoutData:
        camera = relayoutData['scene.camera']
        base_figure.update_layout(
            scene_camera=camera
        )

    # Return the updated figure with the point
    base_figure.update_layout(
        title='Function on Torus',
        scene=dict(
            xaxis=dict(title='X', range=[-5, 5]),
            yaxis=dict(title='Y', range=[-5, 5]),
            zaxis=dict(title='Z', range=[-5, 5]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=1)
        ),
        width=800,  # Increase the width
        height=800  # Increase the height
    )
    return base_figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
