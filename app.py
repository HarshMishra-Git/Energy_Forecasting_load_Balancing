import streamlit as st
import pandas as pd
from src.data_processor import DataProcessor
from src.weather_api import WeatherAPI
from src.visualization import DataVisualizer
from src.data_generator import generate_sample_data
from src.forecasting import EnergyForecaster
from src.load_balancer import LoadBalancer
from src.weather_impact import WeatherImpactAnalyzer
from src.recommendations import EnergyRecommender
from src.pdf_report import PDFReport
from src.onboarding import Onboarding
import io
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Energy Load Forecasting",
    page_icon="⚡",
    layout="wide"
)

# Initialize session state
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'forecaster' not in st.session_state:
    st.session_state.forecaster = EnergyForecaster(model_type='prophet')
if 'load_balancer' not in st.session_state:
    st.session_state.load_balancer = LoadBalancer()
if 'weather_api' not in st.session_state:
    st.session_state.weather_api = WeatherAPI()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = DataVisualizer()
if 'model_type' not in st.session_state:
    st.session_state.model_type = 'prophet'
if 'weather_impact_analyzer' not in st.session_state:
    st.session_state.weather_impact_analyzer = WeatherImpactAnalyzer()
if 'energy_recommender' not in st.session_state:
    st.session_state.energy_recommender = EnergyRecommender()
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'current_tip_index' not in st.session_state:
    st.session_state.current_tip_index = 0

# Onboarding tutorial
onboarding = Onboarding()
onboarding.start()

# Theme toggle
def set_theme():
    if st.session_state.theme == 'light':
        st.session_state.theme = 'dark'
    else:
        st.session_state.theme = 'light'
    st.experimental_set_query_params(theme=st.session_state.theme)

if 'theme' not in st.session_state:
    st.session_state.theme = 'light'

theme_button = st.sidebar.button("Toggle Theme", on_click=set_theme)

# App title and description
st.title("Energy Load Forecasting and Balancing System")
st.markdown("""
## About the Project

Welcome to the Energy Load Forecasting and Balancing System! This project is designed to help you forecast energy demand and optimize load balancing across the smart grid. By leveraging advanced machi[...]

### Key Features:
- **Energy Demand Forecasting:** Use various forecasting models like Prophet, LSTM, and GRU to predict future energy consumption.
- **Weather Impact Analysis:** Understand how weather conditions affect energy consumption and simulate different scenarios.
- **Load Balancing:** Optimize energy distribution across the grid to enhance efficiency and stability.
- **Eco-Friendly Tips:** Get recommendations to reduce energy consumption and promote sustainable practices.
- **Voice-Activated Insights:** Interact with the system using voice commands to get quick insights and analyses.

### How to Use:
1. **Upload Data:** Upload your historical energy consumption data in CSV format.
2. **Process Data:** Preprocess the data and integrate weather information.
3. **Generate Forecasts:** Choose a forecasting model and generate future energy demand predictions.
4. **Analyze Weather Impact:** Simulate temperature changes and analyze their impact on energy consumption.
5. **Optimize Load Balancing:** Calculate load distribution and optimize grid capacity.
6. **Get Recommendations:** Explore eco-friendly tips and voice-activated insights to enhance your energy management strategy.

To get started, navigate through the sidebar options and follow the instructions provided in each section.

""")

# Sidebar - Navigation
st.sidebar.header("Navigation")
navigation = st.sidebar.radio("Go to", ['Home', 'Upload Data', 'Forecasting', 'Weather Impact', 'Consumption Patterns', 'Load Balancing', 'Eco-Friendly Tips'])

# Sidebar - Model Selection
if navigation == 'Home':
    st.sidebar.header("Model Settings")
    model_type = st.sidebar.selectbox(
        "Select Forecasting Model",
        ['prophet', 'lstm', 'gru'],
        help="Prophet: Traditional forecasting, LSTM/GRU: Deep learning models"
    )

    # Update model if changed
    if model_type != st.session_state.model_type:
        try:
            st.session_state.model_type = model_type
            st.session_state.forecaster = EnergyForecaster(model_type=model_type)
            st.sidebar.success(f"Switched to {model_type.upper()} model")
        except Exception as e:
            st.sidebar.error(f"Error switching model: {str(e)}")
            st.session_state.model_type = 'prophet'
            st.session_state.forecaster = EnergyForecaster(model_type='prophet')

    # Demo data option
    st.sidebar.markdown("---")
    if st.sidebar.button("Load Demo Data", key="load_demo_data"):
        try:
            df = generate_sample_data(days=30)
            processed_df = st.session_state.data_processor.preprocess_data(df)
            st.session_state.processed_df = processed_df
            st.sidebar.success("Demo data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading demo data: {str(e)}")

if navigation == 'Upload Data':
    st.sidebar.header("Data Input")
    uploaded_file = st.sidebar.file_uploader("Upload Your Own Data (CSV)", type=['csv'], key="file_uploader")

    if st.sidebar.button("Process Uploaded Data", key="process_uploaded_data"):
        try:
            # Load data
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                processed_df = st.session_state.data_processor.preprocess_data(df)
                st.session_state.processed_df = processed_df
            elif st.session_state.processed_df is not None:
                processed_df = st.session_state.processed_df
            else:
                st.info("Please upload your energy consumption data or use the demo data to begin.")
                st.stop()

            # Ensure the timestamp column is in datetime format
            if not pd.api.types.is_datetime64_any_dtype(processed_df['timestamp']):
                processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])

            # Weather data integration
            st.sidebar.header("Weather Settings")
            lat = st.sidebar.number_input("Latitude", value=40.7128, key="latitude_input")
            lon = st.sidebar.number_input("Longitude", value=-74.0060, key="longitude_input")

            if st.sidebar.button("Fetch Weather Data", key="fetch_weather_data"):
                try:
                    with st.spinner("Fetching weather data..."):
                        if not isinstance(processed_df['timestamp'].iloc[0], pd.Timestamp):
                            processed_df['timestamp'] = pd.to_datetime(processed_df['timestamp'])

                        weather_data = st.session_state.weather_api.get_historical_weather(
                            lat, lon,
                            processed_df['timestamp'].min(),
                            processed_df['timestamp'].max()
                        )

                        if weather_data.empty:
                            st.sidebar.error("Failed to fetch weather data. Please try again later.")
                        else:
                            processed_df = st.session_state.data_processor.create_features(weather_data)
                            st.session_state.processed_df = processed_df
                            st.success("Weather data integrated successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error fetching weather data: {str(e)}")

            # Main content
            if st.session_state.processed_df is not None and not st.session_state.processed_df.empty:
                processed_df = st.session_state.processed_df
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Energy Consumption Trend")
                    consumption_fig = st.session_state.visualizer.plot_consumption_trend(processed_df)
                    st.plotly_chart(consumption_fig, use_container_width=True)

                with col2:
                    st.subheader("Data Statistics")
                    st.write(processed_df.describe())
            else:
                st.warning("No processed data available. Please load the data and fetch weather data if necessary.")
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")

if navigation == 'Forecasting' and st.session_state.processed_df is not None:
    # Forecasting section
    st.header("Energy Demand Forecasting")

    # Model specific settings
    with st.expander("📊 Model Settings", expanded=True):
        if st.session_state.model_type in ['lstm', 'gru']:
            epochs = st.slider("Training Epochs", 10, 100, 50, key="training_epochs")
            batch_size = st.slider("Batch Size", 16, 64, 32, key="batch_size")
            st.info(f"Using {st.session_state.model_type.upper()} neural network for forecasting")
        else:
            st.info("Using Prophet model for forecasting")

    if st.button("Generate Forecast", key="generate_forecast"):
        with st.spinner(f"Training {st.session_state.model_type.upper()} model..."):
            try:
                processed_df = st.session_state.processed_df
                if st.session_state.model_type in ['lstm', 'gru']:
                    if processed_df.empty:
                        st.error("Processed DataFrame is empty. Cannot generate forecast.")
                    else:
                        metrics, results = st.session_state.forecaster.evaluate(
                            processed_df,
                            epochs=epochs,
                            batch_size=batch_size
                        )

                        # Display training progress
                        if 'history' in results and len(results['history']) > 0:
                            st.subheader("Training Progress")
                            st.line_chart(pd.DataFrame(results['history'])['loss'])

                        # Ensure there are enough predictions
                        if 'predictions' in results and len(results['predictions']) > 0:
                            forecast_length = len(results['predictions'])
                            if forecast_length <= len(processed_df):
                                forecast = pd.DataFrame({
                                    'ds': processed_df['timestamp'].iloc[-forecast_length:].values,
                                    'yhat': results['predictions'].flatten(),
                                    'yhat_lower': results['predictions'].flatten() * 0.9,
                                    'yhat_upper': results['predictions'].flatten() * 1.1
                                })
                            else:
                                st.error("Prediction length exceeds available data length. Please check your model and data.")
                        else:
                            st.error("No predictions were made. Please check your model and data.")
                else:
                    metrics, forecast = st.session_state.forecaster.evaluate(processed_df)

                # Display metrics
                st.subheader("Forecast Metrics")
                col3, col4, col5 = st.columns(3)
                col3.metric("MAE", f"{metrics['mae']:.2f}")
                col4.metric("RMSE", f"{metrics['rmse']:.2f}")
                col5.metric("MAPE", f"{metrics['mape']:.2f}%")

                # Plot forecast
                st.subheader("Forecast Results")
                if not forecast.empty:
                    forecast_fig = st.session_state.visualizer.plot_forecast_vs_actual(
                        processed_df, forecast
                    )
                    st.plotly_chart(forecast_fig, use_container_width=True)

                # Export as PDF
                pdf_report = PDFReport()
                pdf_bytes = pdf_report.generate_report(processed_df, forecast, metrics)
                st.download_button(label="Download Forecast Report", data=pdf_bytes, file_name="forecast_report.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"Error generating forecast: {str(e)}")

if navigation == 'Weather Impact' and st.session_state.processed_df is not None:
    # Weather Impact Analysis
    st.header("Weather Impact Analysis")
    if 'temperature' in st.session_state.processed_df.columns:
        with st.expander("📊 Temperature Impact Analysis", expanded=True):
            # Temperature impact slider
            delta_temp = st.slider(
                "Simulate Temperature Change (°C)",
                min_value=-10.0,
                max_value=10.0,
                value=0.0,
                step=0.5,
                key="temperature_change_slider",
                help="Adjust to see how temperature changes affect energy consumption"
            )

            if st.button("Analyze Temperature Impact", key="analyze_temperature_impact"):
                with st.spinner("Analyzing temperature impact..."):
                    # Simulate temperature change
                    simulated_df = st.session_state.weather_impact_analyzer.simulate_temperature_change(
                        st.session_state.processed_df, delta_temp
                    )

                    # Plot comparison
                    impact_fig = st.session_state.visualizer.plot_temperature_impact(
                        st.session_state.processed_df, simulated_df, delta_temp
                    )
                    st.plotly_chart(impact_fig, use_container_width=True)

                    # Display impact metrics
                    impact_analysis = st.session_state.weather_impact_analyzer.analyze_temperature_impact(st.session_state.processed_df)
                    st.info(f"""
                    Temperature Sensitivity: {impact_analysis['sensitivity']:.2f} kWh/°C
                    This means energy consumption changes by approximately {abs(impact_analysis['sensitivity']):.2f} kWh 
                    for each degree Celsius change in temperature.
                    """)

if navigation == 'Consumption Patterns' and st.session_state.processed_df is not None:
    # Energy Consumption Patterns
    st.header("Energy Consumption Patterns")
    with st.expander("🔥 Consumption Heatmap", expanded=True):
        heatmap_fig = st.session_state.visualizer.plot_consumption_heatmap(st.session_state.processed_df, animated=True)
        st.plotly_chart(heatmap_fig, use_container_width=True)

if navigation == 'Load Balancing' and st.session_state.processed_df is not None:
    # Load balancing section
    st.header("Load Balancing Analysis")
    total_capacity = st.number_input(
        "Total Grid Capacity (kWh)",
        value=float(st.session_state.processed_df['energy_consumption'].max() * 1.2),
        key="total_grid_capacity"
    )

    if st.button("Analyze Load Distribution", key="analyze_load_distribution"):
        with st.spinner("Analyzing load distribution..."):
            st.session_state.load_balancer.set_capacity(total_capacity)
            load_distribution = st.session_state.load_balancer.calculate_load_distribution(
                st.session_state.processed_df['energy_consumption'].values
            )

            load_distribution_df = pd.DataFrame({
                'time_point': np.arange(len(load_distribution['load_percentage'])),
                'load_percentage': load_distribution['load_percentage'],
                'suggested_redistribution': load_distribution['suggested_redistribution']
            })

            load_fig = st.session_state.visualizer.plot_load_distribution(load_distribution_df)
            st.plotly_chart(load_fig, use_container_width=True)

            # Export as PDF
            pdf_report = PDFReport()
            pdf_bytes = pdf_report.generate_report(st.session_state.processed_df, load_distribution_df, metrics=None, report_type="load_balancing")
            st.download_button(label="Download Load Balancing Report", data=pdf_bytes, file_name="load_balancing_report.pdf", mime="application/pdf")

if navigation == 'Eco-Friendly Tips':
    # Eco-Friendly Tips Carousel
    st.header("Eco-Friendly Tips")
    tips = st.session_state.energy_recommender.get_eco_friendly_tips()
    current_tip_index = st.session_state.current_tip_index

    st.markdown(
        f"""
        <div style="text-align: center; background-color: #f0f0f0; padding: 20px; border-radius: 10px; font-size: 18px; color: black;">
            {tips[current_tip_index]}
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        if st.button("Previous", key="previous_tip"):
            st.session_state.current_tip_index = (current_tip_index - 1) % len(tips)
    with col3:
        if st.button("Next", key="next_tip"):
            st.session_state.current_tip_index = (current_tip_index + 1) % len(tips)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by [Harsh Mishra](https://harshmishra-git.github.io/MY-Portfolio/) (aka Data Scientist)")
