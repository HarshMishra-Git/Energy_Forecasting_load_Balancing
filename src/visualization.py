import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

class DataVisualizer:
    @staticmethod
    def plot_consumption_trend(df, animated=False):
        """Plot energy consumption trend"""
        fig = px.line(
            df,
            x='timestamp',
            y='energy_consumption',
            title='Energy Consumption Over Time'
        )
        if animated:
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    )
                )
            )
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Energy Consumption (kWh)",
            template="plotly_white"
        )
        return fig

    @staticmethod
    def plot_consumption_heatmap(df, animated=False):
        """Create animated heatmap of energy consumption patterns"""
        # Convert timestamp to hour and day
        df_heatmap = df.copy()
        df_heatmap['hour'] = df_heatmap['timestamp'].dt.hour
        df_heatmap['day'] = df_heatmap['timestamp'].dt.date

        # Pivot data for heatmap
        pivot_table = df_heatmap.pivot_table(
            values='energy_consumption',
            index='day',
            columns='hour',
            aggfunc='mean'
        )

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns,
            y=pivot_table.index,
            colorscale='Viridis',
            colorbar=dict(title='Energy Consumption (kWh)')
        ))

        if animated:
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(
                        visible=True
                    )
                )
            )

        fig.update_layout(
            title='Energy Consumption Patterns',
            xaxis_title='Hour of Day',
            yaxis_title='Date',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_temperature_impact(actual_df, simulated_df, delta_temp):
        """Plot temperature impact comparison"""
        fig = go.Figure()

        # Plot actual consumption
        fig.add_trace(go.Scatter(
            x=actual_df['timestamp'],
            y=actual_df['energy_consumption'],
            name='Actual Consumption',
            line=dict(color='#0366D6')
        ))

        # Plot simulated consumption
        fig.add_trace(go.Scatter(
            x=simulated_df['timestamp'],
            y=simulated_df['energy_consumption'],
            name=f'Simulated (Δ{delta_temp}°C)',
            line=dict(color='#DC3545', dash='dash')
        ))

        fig.update_layout(
            title='Temperature Impact Analysis',
            xaxis_title='Time',
            yaxis_title='Energy Consumption (kWh)',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_forecast_vs_actual(actual, forecast):
        """Plot forecast vs actual values"""
        fig = go.Figure()

        # Add actual values
        fig.add_trace(go.Scatter(
            x=actual['timestamp'],
            y=actual['energy_consumption'],
            name='Actual',
            line=dict(color='#0366D6')
        ))

        # Add forecast values
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Forecast',
            line=dict(color='#28A745')
        ))

        # Add confidence intervals
        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='rgba(40, 167, 69, 0.1)',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='rgba(40, 167, 69, 0.1)',
            name='95% Confidence Interval'
        ))

        fig.update_layout(
            title='Forecast vs Actual Energy Consumption',
            xaxis_title='Time',
            yaxis_title='Energy Consumption (kWh)',
            template='plotly_white'
        )

        return fig

    @staticmethod
    def plot_load_distribution(load_data):
        """Plot load distribution analysis"""
        # Convert time points to numpy array
        time_points = np.arange(len(load_data['load_percentage']))

        fig = make_subplots(
            rows=2, 
            cols=1,
            subplot_titles=(
                'Load Distribution Over Time',
                'Suggested Load Redistribution'
            )
        )

        # Load percentage
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=load_data['load_percentage'],
                name='Load Percentage',
                line=dict(color='#0366D6'),
                hovertemplate='Time Point: %{x}<br>Load: %{y:.1%}<extra></extra>'
            ),
            row=1, col=1
        )

        # Add threshold lines
        fig.add_hline(
            y=0.85, 
            line_dash="dash", 
            line_color="red",
            annotation_text="High Load Threshold (85%)",
            row=1, col=1
        )
        fig.add_hline(
            y=0.15, 
            line_dash="dash", 
            line_color="orange",
            annotation_text="Low Load Threshold (15%)",
            row=1, col=1
        )

        # Suggested redistribution
        fig.add_trace(
            go.Bar(
                x=time_points,
                y=load_data['suggested_redistribution'],
                name='Suggested Redistribution',
                marker_color=np.where(
                    load_data['suggested_redistribution'] >= 0,
                    '#28A745',  # Green for positive values
                    '#DC3545'   # Red for negative values
                ),
                hovertemplate='Time Point: %{x}<br>Adjustment: %{y:.1f} kWh<extra></extra>'
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=800,
            title='Load Distribution Analysis',
            template='plotly_white',
            showlegend=True
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Load Percentage", row=1, col=1)
        fig.update_yaxes(title_text="Load Adjustment (kWh)", row=2, col=1)

        # Update x-axes labels
        fig.update_xaxes(title_text="Time Points", row=1, col=1)
        fig.update_xaxes(title_text="Time Points", row=2, col=1)

        return fig