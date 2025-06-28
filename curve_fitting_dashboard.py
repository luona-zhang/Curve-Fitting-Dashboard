import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # REQUIRED for gunicorn

app.title = "Curve Fitting Dashboard"

# Initial sample data
initial_data = [
    {"id": 1, "x": 1, "y": 2},
    {"id": 2, "x": 2, "y": 4.1},
    {"id": 3, "x": 3, "y": 8.9},
    {"id": 4, "x": 4, "y": 16.2}
]

# Define the app layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1([
            "📈 Curve Fitting Dashboard",
        ], className="header-title"),
        html.P("Interactive data visualization with real-time curve fitting", 
               className="header-subtitle")
    ], className="header"),
    
    # Main content
    html.Div([
        # Left sidebar
        html.Div([
            # Add Point Section
            html.Div([
                html.H3("➕ Add Point", className="section-title"),
                html.Div([
                    dcc.Input(
                        id="input-x",
                        type="number",
                        placeholder="X coordinate",
                        className="input-field",
                        style={"width": "48%", "marginRight": "4%"}
                    ),
                    dcc.Input(
                        id="input-y",
                        type="number",
                        placeholder="Y coordinate",
                        className="input-field",
                        style={"width": "48%"}
                    )
                ], style={"marginBottom": "10px"}),
                html.Button("Add Point", id="add-point-btn", className="btn-primary"),
            ], className="section-card"),
            
            # Curve Fitting Options
            html.Div([
                html.H3("🔧 Fitting Options", className="section-title"),
                html.Label("Curve Type:", className="input-label"),
                dcc.Dropdown(
                    id="curve-type",
                    options=[
                        {"label": "Linear", "value": "linear"},
                        {"label": "Polynomial", "value": "polynomial"},
                        {"label": "Exponential", "value": "exponential"},
                        {"label": "Power", "value": "power"}
                    ],
                    value="polynomial",
                    className="dropdown"
                ),
                html.Div([
                    html.Label(f"Polynomial Degree:", className="input-label"),
                    dcc.Slider(
                        id="poly-degree",
                        min=1,
                        max=6,
                        step=1,
                        value=2,
                        marks={i: str(i) for i in range(1, 7)},
                        className="slider"
                    )
                ], id="poly-degree-container", style={"marginTop": "15px"})
            ], className="section-card"),
            
            # Metrics
            html.Div([
                html.H3("📊 Fit Metrics", className="section-title"),
                html.Div(id="metrics-display")
            ], className="section-card"),
            
            # Equation
            html.Div([
                html.H3("🧮 Equation", className="section-title"),
                html.Div(id="equation-display", className="equation-box")
            ], className="section-card"),
            
        ], className="sidebar"),
        
        # Right content area
        html.Div([
            # Chart
            html.Div([
                html.Div([
                    html.H3("📈 Visualization", className="section-title"),
                    html.Button("🔄 Reset", id="reset-btn", className="btn-secondary")
                ], className="chart-header"),
                dcc.Graph(id="main-chart", className="chart")
            ], className="chart-card"),
            
            # Data Table
            html.Div([
                html.H3("📋 Data Points", className="section-title"),
                dash_table.DataTable(
                    id="data-table",
                    columns=[
                        {"name": "X", "id": "x", "type": "numeric", "editable": True},
                        {"name": "Y", "id": "y", "type": "numeric", "editable": True},
                        {"name": "Actions", "id": "actions", "presentation": "markdown"}
                    ],
                    data=[{**point, "actions": "🗑️"} for point in initial_data],
                    editable=True,
                    row_deletable=True,
                    style_cell={"textAlign": "center", "backgroundColor": "rgba(255,255,255,0.1)", "color": "white"},
                    style_header={"backgroundColor": "rgba(255,255,255,0.2)", "fontWeight": "bold"},
                    style_data_conditional=[
                        {
                            "if": {"row_index": "odd"},
                            "backgroundColor": "rgba(255,255,255,0.05)"
                        }
                    ]
                )
            ], className="table-card")
        ], className="main-content")
    ], className="container"),
    
    # Store components for data persistence
    dcc.Store(id="points-store", data=initial_data),
    dcc.Store(id="point-counter", data=len(initial_data))
], className="app-container")

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                color: white;
            }
            
            .app-container {
                min-height: 100vh;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .header-title {
                font-size: 2.5rem;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            
            .header-subtitle {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .container {
                display: grid;
                grid-template-columns: 350px 1fr;
                gap: 25px;
                max-width: 1400px;
                margin: 0 auto;
            }
            
            .sidebar {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .main-content {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            
            .section-card, .chart-card, .table-card {
                background: rgba(255, 255, 255, 0.1);
                backdrop-filter: blur(10px);
                border-radius: 15px;
                padding: 20px;
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .section-title {
                margin-bottom: 15px;
                font-size: 1.2rem;
            }
            
            .input-field {
                padding: 10px;
                border-radius: 8px;
                border: 1px solid rgba(255, 255, 255, 0.3);
                background: rgba(255, 255, 255, 0.2);
                color: white;
                font-size: 14px;
            }
            
            .input-field::placeholder {
                color: rgba(255, 255, 255, 0.6);
            }
            
            .input-label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
            }
            
            .btn-primary {
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: 600;
                width: 100%;
                transition: transform 0.2s;
            }
            
            .btn-primary:hover {
                transform: translateY(-2px);
            }
            
            .btn-secondary {
                background: rgba(255, 255, 255, 0.2);
                color: white;
                border: 1px solid rgba(255, 255, 255, 0.3);
                padding: 8px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 12px;
            }
            
            .dropdown > div {
                background: rgba(255, 255, 255, 0.2) !important;
                border: 1px solid rgba(255, 255, 255, 0.3) !important;
                color: white !important;
            }
            
            .slider .rc-slider-track {
                background: linear-gradient(135deg, #667eea, #764ba2) !important;
            }
            
            .slider .rc-slider-handle {
                border-color: #667eea !important;
            }
            
            .chart-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
            }
            
            .chart {
                height: 400px;
            }
            
            .equation-box {
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 14px;
                color: #64ffda;
                word-break: break-all;
            }
            
            .metric-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding: 5px 0;
            }
            
            .metric-label {
                opacity: 0.8;
            }
            
            .metric-value {
                font-family: 'Courier New', monospace;
                font-weight: bold;
            }
            
            @media (max-width: 768px) {
                .container {
                    grid-template-columns: 1fr;
                }
                
                .header-title {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Curve fitting functions
def fit_linear(x, y):
    """Linear regression"""
    model = LinearRegression()
    X = np.array(x).reshape(-1, 1)
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    return {
        'equation': f'y = {slope:.3f}x + {intercept:.3f}',
        'model': model,
        'type': 'linear'
    }

def fit_polynomial(x, y, degree):
    """Polynomial regression"""
    poly_features = PolynomialFeatures(degree=degree)
    X = np.array(x).reshape(-1, 1)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Generate equation string
    coeffs = model.coef_
    intercept = model.intercept_
    equation_parts = [f'{intercept:.3f}']
    
    for i in range(1, len(coeffs)):
        coeff = coeffs[i]
        if abs(coeff) > 1e-10:
            sign = ' + ' if coeff >= 0 else ' - '
            if i == 1:
                equation_parts.append(f'{sign}{abs(coeff):.3f}x')
            else:
                equation_parts.append(f'{sign}{abs(coeff):.3f}x^{i}')
    
    equation = 'y = ' + ''.join(equation_parts)
    
    return {
        'equation': equation,
        'model': model,
        'poly_features': poly_features,
        'type': 'polynomial'
    }

def fit_exponential(x, y):
    """Exponential regression y = ae^(bx)"""
    try:
        # Filter positive y values
        valid_indices = np.array(y) > 0
        if not any(valid_indices):
            return fit_linear(x, y)
        
        x_valid = np.array(x)[valid_indices]
        y_valid = np.array(y)[valid_indices]
        log_y = np.log(y_valid)
        
        model = LinearRegression()
        X = x_valid.reshape(-1, 1)
        model.fit(X, log_y)
        
        b = model.coef_[0]
        log_a = model.intercept_
        a = np.exp(log_a)
        
        return {
            'equation': f'y = {a:.3f}e^({b:.3f}x)',
            'a': a,
            'b': b,
            'type': 'exponential'
        }
    except:
        return fit_linear(x, y)

def fit_power(x, y):
    """Power regression y = ax^b"""
    try:
        # Filter positive values
        valid_indices = (np.array(x) > 0) & (np.array(y) > 0)
        if not any(valid_indices):
            return fit_linear(x, y)
        
        x_valid = np.array(x)[valid_indices]
        y_valid = np.array(y)[valid_indices]
        log_x = np.log(x_valid)
        log_y = np.log(y_valid)
        
        model = LinearRegression()
        X = log_x.reshape(-1, 1)
        model.fit(X, log_y)
        
        b = model.coef_[0]
        log_a = model.intercept_
        a = np.exp(log_a)
        
        return {
            'equation': f'y = {a:.3f}x^{b:.3f}',
            'a': a,
            'b': b,
            'type': 'power'
        }
    except:
        return fit_linear(x, y)

def predict_values(fit_result, x_range):
    """Generate predictions for plotting"""
    if fit_result['type'] == 'linear':
        X = np.array(x_range).reshape(-1, 1)
        return fit_result['model'].predict(X)
    elif fit_result['type'] == 'polynomial':
        X = np.array(x_range).reshape(-1, 1)
        X_poly = fit_result['poly_features'].transform(X)
        return fit_result['model'].predict(X_poly)
    elif fit_result['type'] == 'exponential':
        return fit_result['a'] * np.exp(fit_result['b'] * np.array(x_range))
    elif fit_result['type'] == 'power':
        x_arr = np.array(x_range)
        return np.where(x_arr > 0, fit_result['a'] * np.power(x_arr, fit_result['b']), 0)

def calculate_metrics(y_true, y_pred):
    """Calculate fit metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {'mae': mae, 'rmse': rmse, 'r2': max(0, r2)}

# Callbacks
@app.callback(
    [Output('points-store', 'data'),
     Output('point-counter', 'data'),
     Output('input-x', 'value'),
     Output('input-y', 'value')],
    [Input('add-point-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State('input-x', 'value'),
     State('input-y', 'value'),
     State('points-store', 'data'),
     State('point-counter', 'data')]
)
def update_points(add_clicks, reset_clicks, x_val, y_val, current_points, counter):
    ctx = callback_context
    if not ctx.triggered:
        return current_points, counter, None, None
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if button_id == 'reset-btn':
        return initial_data, len(initial_data), None, None
    elif button_id == 'add-point-btn' and x_val is not None and y_val is not None:
        new_point = {"id": counter + 1, "x": x_val, "y": y_val}
        updated_points = current_points + [new_point]
        return updated_points, counter + 1, None, None
    
    return current_points, counter, None, None

@app.callback(
    Output('points-store', 'data', allow_duplicate=True),
    Input('data-table', 'data'),
    prevent_initial_call=True
)
def update_points_from_table(table_data):
    # Filter out rows that don't have both x and y values
    valid_points = []
    for i, row in enumerate(table_data):
        if row.get('x') is not None and row.get('y') is not None:
            valid_points.append({
                "id": i + 1,
                "x": float(row['x']),
                "y": float(row['y'])
            })
    return valid_points

@app.callback(
    Output('data-table', 'data'),
    Input('points-store', 'data')
)
def update_table(points_data):
    return [{**point, "actions": "🗑️"} for point in points_data]

@app.callback(
    Output('poly-degree-container', 'style'),
    Input('curve-type', 'value')
)
def toggle_poly_degree(curve_type):
    if curve_type == 'polynomial':
        return {"marginTop": "15px", "display": "block"}
    return {"display": "none"}

@app.callback(
    [Output('main-chart', 'figure'),
     Output('equation-display', 'children'),
     Output('metrics-display', 'children')],
    [Input('points-store', 'data'),
     Input('curve-type', 'value'),
     Input('poly-degree', 'value')]
)
def update_chart_and_metrics(points_data, curve_type, poly_degree):
    if len(points_data) < 2:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Add at least 2 points to see curve fitting",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0.1)"
        )
        return empty_fig, "Need at least 2 points", html.Div("No metrics available")
    
    # Extract x and y values
    x_data = [point['x'] for point in points_data]
    y_data = [point['y'] for point in points_data]
    
    # Fit curve based on selected type
    if curve_type == 'linear':
        fit_result = fit_linear(x_data, y_data)
    elif curve_type == 'polynomial':
        fit_result = fit_polynomial(x_data, y_data, poly_degree)
    elif curve_type == 'exponential':
        fit_result = fit_exponential(x_data, y_data)
    elif curve_type == 'power':
        fit_result = fit_power(x_data, y_data)
    
    # Generate curve points for plotting
    x_min, x_max = min(x_data) - 1, max(x_data) + 1
    x_curve = np.linspace(x_min, x_max, 100)
    y_curve = predict_values(fit_result, x_curve)
    
    # Create figure
    fig = go.Figure()
    
    # Add data points
    fig.add_trace(go.Scatter(
        x=x_data,
        y=y_data,
        mode='markers',
        marker=dict(size=12, color='#f59e0b', line=dict(width=2, color='white')),
        name='Data Points'
    ))
    
    # Add fitted curve
    fig.add_trace(go.Scatter(
        x=x_curve,
        y=y_curve,
        mode='lines',
        line=dict(color='#8b5cf6', width=3),
        name='Fitted Curve'
    ))
    
    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0.1)",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.5)"),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    # Calculate metrics
    if fit_result['type'] in ['linear', 'polynomial']:
        if fit_result['type'] == 'linear':
            X = np.array(x_data).reshape(-1, 1)
            y_pred = fit_result['model'].predict(X)
        else:
            X = np.array(x_data).reshape(-1, 1)
            X_poly = fit_result['poly_features'].transform(X)
            y_pred = fit_result['model'].predict(X_poly)
    else:
        y_pred = predict_values(fit_result, x_data)
    
    metrics = calculate_metrics(y_data, y_pred)
    
    # Format metrics display
    metrics_div = html.Div([
        html.Div([
            html.Span("R²:", className="metric-label"),
            html.Span(f"{metrics['r2']:.4f}", className="metric-value", style={"color": "#4ade80"})
        ], className="metric-item"),
        html.Div([
            html.Span("MAE:", className="metric-label"),
            html.Span(f"{metrics['mae']:.4f}", className="metric-value", style={"color": "#60a5fa"})
        ], className="metric-item"),
        html.Div([
            html.Span("RMSE:", className="metric-label"),
            html.Span(f"{metrics['rmse']:.4f}", className="metric-value", style={"color": "#fb7185"})
        ], className="metric-item")
    ])
    
    return fig, fit_result['equation'], metrics_div

if __name__ == '__main__':
    app.run()
