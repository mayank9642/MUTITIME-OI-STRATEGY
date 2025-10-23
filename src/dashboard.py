import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
import os
import logging
import datetime
import time
import pytz
import json
import plotly.express as px
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    filename='logs/dashboard.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

from src.strategy import OpenInterestStrategy, run_strategy
from src.config import load_config
from src.token_helper import ensure_valid_token
from src.fyers_api_utils import get_fyers_client

# Ensure logs directory exists
os.makedirs('logs', exist_ok=True)

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "OI Breakout Strategy Dashboard"

# Initialize strategy
strategy = OpenInterestStrategy()

# Format for dates and times
date_fmt = "%Y-%m-%d"
time_fmt = "%H:%M:%S"

# Define app layout
app.layout = html.Div([
    html.Div([
        html.H1("OI Breakout Strategy Dashboard", className='dashboard-title'),
        html.Div(id='live-clock', className='clock-display'),
    ], className='header'),
    
    # Tabs for different sections
    dcc.Tabs([
        dcc.Tab(label='OI Analysis', children=[
            html.Div([
                html.Div([
                    html.H3("Highest OI Strikes"),
                    html.Div([
                        html.Div([
                            html.H4("Put OI"),
                            html.Div(id='put-oi-info', className='oi-display')
                        ], className='oi-container'),
                        html.Div([
                            html.H4("Call OI"),
                            html.Div(id='call-oi-info', className='oi-display')
                        ], className='oi-container'),
                    ], className='oi-flex-container'),
                ], className='card'),
                
                html.Div([
                    html.H3("OI Distribution"),
                    dcc.Graph(id='oi-distribution-chart')
                ], className='card large-card'),
                
                html.Div([
                    html.H3("Breakout Monitor"),
                    html.Div(id='breakout-status', className='status-display')
                ], className='card'),
            ], className='grid-layout')
        ]),
        
        dcc.Tab(label='Trade History', children=[
            html.Div([
                html.Div([
                    html.H3("Recent Trades"),
                    html.Div(id='trade-history-table-container')
                ], className='card large-card'),
                
                html.Div([
                    html.H3("Trade Performance"),
                    dcc.Graph(id='trade-performance-chart')
                ], className='card large-card'),
            ], className='grid-layout')
        ]),
        
        dcc.Tab(label='System Status', children=[
            html.Div([
                html.Div([
                    html.H3("Fyers API Status"),
                    html.Div(id='api-status', className='status-display')
                ], className='card'),
                
                html.Div([
                    html.H3("Strategy Settings"),
                    html.Div(id='strategy-settings', className='settings-display')
                ], className='card large-card'),
                
                html.Div([
                    html.H3("System Log"),
                    html.Div(id='log-display', className='log-display')
                ], className='card large-card'),
            ], className='grid-layout')
        ]),
    ]),
    
    # Background refresher
    dcc.Interval(
        id='interval-component',
        interval=2000,  # 2 seconds refresh
        n_intervals=0
    )
], className='dashboard-container')


# Callback to update the clock
@app.callback(
    Output('live-clock', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_clock(n):
    ist_now = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
    return html.P(f"Current IST: {ist_now.strftime('%Y-%m-%d %H:%M:%S')}")


# Callback to update OI information
@app.callback(
    [Output('put-oi-info', 'children'), 
     Output('call-oi-info', 'children'),
     Output('breakout-status', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_oi_info(n):
    try:
        # Ensure we have the latest data
        option_chain = strategy.identify_high_oi_strikes()
        
        # Formatted PUT OI info
        put_info = [
            html.P(f"Strike: {strategy.highest_put_oi_strike}", className='oi-text'),
            html.P(f"Premium: {strategy.put_premium_at_9_20}", className='oi-text'),
            html.P(f"Breakout Level: {strategy.put_breakout_level}", className='oi-text')
        ]
        
        # Formatted CALL OI info
        call_info = [
            html.P(f"Strike: {strategy.highest_call_oi_strike}", className='oi-text'),
            html.P(f"Premium: {strategy.call_premium_at_9_20}", className='oi-text'),
            html.P(f"Breakout Level: {strategy.call_breakout_level}", className='oi-text')
        ]
        
        # Breakout status
        result = strategy.monitor_for_breakout()
        if result:
            breakout_status = [
                html.P(f"BREAKOUT DETECTED: {strategy.active_trade['symbol']}", className='alert-text'),
                html.P(f"Entry Price: {strategy.active_trade['entry_price']}", className='info-text'),
                html.P(f"Stoploss: {strategy.active_trade['stoploss']}", className='info-text'),
                html.P(f"Target: {strategy.active_trade['target']}", className='info-text')
            ]
        else:
            breakout_status = [
                html.P("No breakout detected", className='info-text'),
                html.P(f"PUT Premium: {strategy.put_premium_at_9_20} (Level: {strategy.put_breakout_level})", className='info-text'),
                html.P(f"CALL Premium: {strategy.call_premium_at_9_20} (Level: {strategy.call_breakout_level})", className='info-text')
            ]
        
        return put_info, call_info, breakout_status
    except Exception as e:
        logging.error(f"Error updating OI info: {str(e)}")
        return [html.P("Error loading data", className='error-text')], [html.P("Error loading data", className='error-text')], [html.P("Error checking breakout", className='error-text')]


# Callback to update OI distribution chart
@app.callback(
    Output('oi-distribution-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_oi_chart(n):
    try:
        # Get option chain data
        result = strategy.identify_high_oi_strikes()
        
        # Create two bar charts, one for puts and one for calls
        put_data = []
        call_data = []
        
        option_chain = strategy.get_option_chain()
        if option_chain is not None and not option_chain.empty:
            # Filter for PUTs and CALLs
            put_data = option_chain[option_chain['option_type'] == 'PE']
            call_data = option_chain[option_chain['option_type'] == 'CE']
            
            # Sort by strike price
            put_data = put_data.sort_values('strikePrice')
            call_data = call_data.sort_values('strikePrice')
        
        # Create the figure
        fig = go.Figure()
        
        # Add PUT OI as negative bars
        if not put_data.empty:
            fig.add_trace(go.Bar(
                x=put_data['strikePrice'],
                y=-put_data['openInterest']/1000,  # Divide by 1000 for better scaling
                name='PUT OI',
                marker_color='red'
            ))
        
        # Add CALL OI as positive bars
        if not call_data.empty:
            fig.add_trace(go.Bar(
                x=call_data['strikePrice'],
                y=call_data['openInterest']/1000,  # Divide by 1000 for better scaling
                name='CALL OI',
                marker_color='green'
            ))
        
        # Update layout
        fig.update_layout(
            title='Option Open Interest Distribution (in thousands)',
            xaxis_title='Strike Price',
            yaxis_title='Open Interest (x1000)',
            barmode='relative',
            bargap=0.1,
            template='plotly_dark'
        )
        
        return fig
    except Exception as e:
        logging.error(f"Error updating OI chart: {str(e)}")
        # Return empty figure in case of error
        return go.Figure()


# Callback to update trade history table
@app.callback(
    Output('trade-history-table-container', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_trade_history(n):
    try:
        if os.path.exists('logs/trade_history.csv'):
            df = pd.read_csv('logs/trade_history.csv')
            
            if not df.empty:
                # Format columns for display
                df['entry_time'] = df['entry_time'].fillna('N/A')
                df['exit_time'] = df['exit_time'].fillna('N/A')
                df['pnl'] = df['pnl'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else 'N/A')
                df['paper_trade'] = df['paper_trade'].apply(lambda x: 'Paper' if x else 'Live')
                
                # Create a DataTable
                table = dash_table.DataTable(
                    id='trade-table',
                    columns=[{"name": col.replace('_', ' ').title(), "id": col} for col in df.columns],
                    data=df.to_dict('records'),
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'fontWeight': 'bold'
                    },
                    style_cell={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white',
                        'textAlign': 'left',
                        'padding': '8px'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(40, 40, 40)'
                        }
                    ],
                    page_size=10
                )
                return table
            else:
                return html.P("No trade history available", className='info-text')
        else:
            return html.P("No trade history file found", className='info-text')
    except Exception as e:
        logging.error(f"Error updating trade history: {str(e)}")
        return html.P(f"Error loading trade history: {str(e)}", className='error-text')


# Callback to update performance chart
@app.callback(
    Output('trade-performance-chart', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_performance_chart(n):
    try:
        if os.path.exists('logs/trade_performance.csv'):
            df = pd.read_csv('logs/trade_performance.csv')
            
            if not df.empty:
                # Create a cumulative P&L chart
                df['cumulative_pnl'] = df['pnl'].cumsum()
                
                fig = px.line(
                    df, 
                    x=df.index, 
                    y='cumulative_pnl',
                    title='Cumulative P&L',
                    labels={'index': 'Trade Number', 'cumulative_pnl': 'Cumulative P&L'}
                )
                
                fig.update_traces(line=dict(color='green'))
                fig.update_layout(template='plotly_dark')
                
                return fig
            else:
                # Return empty figure
                fig = go.Figure()
                fig.update_layout(
                    title='No trade performance data available',
                    template='plotly_dark'
                )
                return fig
        else:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title='No trade performance file found',
                template='plotly_dark'
            )
            return fig
    except Exception as e:
        logging.error(f"Error updating performance chart: {str(e)}")
        # Return empty figure in case of error
        fig = go.Figure()
        fig.update_layout(
            title=f'Error: {str(e)}',
            template='plotly_dark'
        )
        return fig


# Callback to update API status
@app.callback(
    Output('api-status', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_api_status(n):
    try:
        # Check if we have a valid token
        access_token = ensure_valid_token()
        
        if access_token:
            token_status = "VALID"
            status_class = "status-ok"
        else:
            token_status = "EXPIRED OR INVALID"
            status_class = "status-error"
        
        # Check if we can connect to Fyers API
        try:
            fyers = get_fyers_client()
            if fyers:
                api_status = "CONNECTED"
                api_class = "status-ok"
            else:
                api_status = "DISCONNECTED"
                api_class = "status-error"
        except:
            api_status = "ERROR"
            api_class = "status-error"
        
        return [
            html.P([
                "Token Status: ",
                html.Span(token_status, className=status_class)
            ]),
            html.P([
                "API Connection: ",
                html.Span(api_status, className=api_class)
            ])
        ]
    except Exception as e:
        logging.error(f"Error updating API status: {str(e)}")
        return html.P(f"Error checking API status: {str(e)}", className='error-text')


# Callback to display strategy settings
@app.callback(
    Output('strategy-settings', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_strategy_settings(n):
    try:
        config = load_config()
        strategy_config = config.get('strategy', {})
        
        settings = [
            html.P(f"Analysis Time: {strategy_config.get('analysis_time', 'N/A')}", className='setting-item'),
            html.P(f"Breakout Percentage: {strategy_config.get('breakout_pct', 'N/A')}%", className='setting-item'),
            html.P(f"Max Holding Time: {strategy_config.get('max_holding_minutes', 'N/A')} minutes", className='setting-item'),
            html.P(f"Risk-Reward Ratio: {strategy_config.get('risk_reward_ratio', 'N/A')}", className='setting-item'),
            html.P(f"Stoploss Percentage: {strategy_config.get('stoploss_pct', 'N/A')}%", className='setting-item'),
            html.P(f"Use Trailing Stop: {'Yes' if strategy_config.get('use_trailing_stop', False) else 'No'}", className='setting-item'),
            html.P(f"Trailing Stop Percentage: {strategy_config.get('trailing_stop_pct', 'N/A')}%", className='setting-item')
        ]
        
        return settings
    except Exception as e:
        logging.error(f"Error updating strategy settings: {str(e)}")
        return html.P(f"Error loading settings: {str(e)}", className='error-text')


# Callback to display system logs
@app.callback(
    Output('log-display', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_logs(n):
    try:
        # Read the last few lines of the strategy log
        log_path = 'logs/strategy.log'
        if os.path.exists(log_path):
            with open(log_path, 'r') as file:
                # Read the last 20 lines
                lines = file.readlines()[-20:]
                
                # Format logs
                log_entries = [html.P(line.strip(), className='log-entry') for line in lines]
                return log_entries
        else:
            return html.P("Log file not found", className='error-text')
    except Exception as e:
        logging.error(f"Error updating logs: {str(e)}")
        return html.P(f"Error reading logs: {str(e)}", className='error-text')


# Add a method to OpenInterestStrategy to return the option chain
def get_option_chain(self):
    """Add this method to the OpenInterestStrategy class to get the current option chain"""
    try:
        return get_nifty_option_chain()
    except Exception as e:
        logging.error(f"Error getting option chain: {str(e)}")
        return None
        
# Add the method to the class
OpenInterestStrategy.get_option_chain = get_option_chain


# Custom CSS for the dashboard
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #121212;
                color: #ffffff;
                margin: 0;
                padding: 0;
            }
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
            }
            .dashboard-title {
                margin: 0;
                color: #50C878;
            }
            .clock-display {
                font-size: 18px;
                font-weight: bold;
            }
            .grid-layout {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }
            .card {
                background-color: #1E1E1E;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            }
            .large-card {
                grid-column: span 2;
            }
            .oi-flex-container {
                display: flex;
                justify-content: space-between;
            }
            .oi-container {
                flex: 1;
                margin: 0 10px;
                padding: 15px;
                background-color: #2A2A2A;
                border-radius: 8px;
            }
            .oi-display {
                font-size: 16px;
            }
            .oi-text {
                margin: 5px 0;
            }
            .status-display {
                padding: 10px;
                background-color: #2A2A2A;
                border-radius: 8px;
            }
            .settings-display {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 10px;
            }
            .setting-item {
                margin: 5px 0;
                padding: 8px;
                background-color: #2A2A2A;
                border-radius: 4px;
            }
            .log-display {
                height: 300px;
                overflow-y: auto;
                background-color: #2A2A2A;
                padding: 10px;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
            }
            .log-entry {
                margin: 2px 0;
                white-space: pre-wrap;
            }
            .status-ok {
                color: #4CAF50;
                font-weight: bold;
            }
            .status-error {
                color: #F44336;
                font-weight: bold;
            }
            .info-text {
                color: #B3E5FC;
            }
            .alert-text {
                color: #FF9800;
                font-weight: bold;
                font-size: 18px;
            }
            .error-text {
                color: #F44336;
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


if __name__ == '__main__':
    # Start the dashboard
    try:
        logging.info("Starting OI Strategy Dashboard...")
        app.run_server(debug=False, host='0.0.0.0', port=8050)
    except Exception as e:
        logging.error(f"Error starting dashboard: {str(e)}")
