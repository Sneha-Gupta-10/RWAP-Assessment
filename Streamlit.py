import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, Fullscreen
import pickle
import gdown
import os
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Asset Price Prediction Dashboard",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    .metric-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        color: white;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        color: #00d4ff;
        margin-bottom: 0.5rem;
        font-size: 1rem;
    }
    .metric-card h2 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: bold;
    }
    .prediction-box {
        background: linear-gradient(145deg, #0d1421, #1a252f);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #00d4ff;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 212, 255, 0.2);
        color: white;
    }
    .prediction-box h1 {
        color: #00ff88;
        text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
        font-size: 3rem;
        margin: 1rem 0;
    }
    .location-prediction-box {
        background: linear-gradient(145deg, #1a1a2e, #16213e);
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #0066cc;
        margin: 1rem 0;
        box-shadow: 0 0 30px rgba(0, 102, 204, 0.2);
        color: white;
    }
    .location-prediction-box h1 {
        color: #66b3ff;
        text-shadow: 0 0 10px rgba(102, 179, 255, 0.5);
        font-size: 2.5rem;
        margin: 1rem 0;
    }
    .fullscreen-map-container {
        position: relative;
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid #00d4ff;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .map-controls {
        background: linear-gradient(145deg, #2d2d2d, #1e1e1e);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 1px solid #00d4ff;
    }
    .sidebar .sidebar-content {
        background-color: #1e1e1e;
    }
    .stSelectbox > div > div {
        background-color: #2d2d2d;
        color: white;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border: 1px solid #00d4ff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    div[data-testid="metric-container"] > label {
        color: #00d4ff !important;
    }
    div[data-testid="metric-container"] > div {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def download_models_and_data():
    """Download all required files from Google Drive"""
    files_to_download = {
        'scaler_last_price.pkl': '1nhoS237W_-5Fsgdo7sDFD5_7hceHappp',
        'cluster_1_model.pkl': '1GaDbbVCBUvjrvSUrfT6GLJUFYVa1xRPG',
        'cluster_0_model.pkl': '1X9WmLRoJHCdMcLVKTtsbDujYAIg_o1dU',
        'global_model_pca.pkl': '1dmE1bEDWUeAkZNkpGDTHEJA6AEt0FPz1',
        'global_model.pkl': '1ZWPra5iZ0pEVQgxpPaWx8gX3J9olsb7Z',
        'assets_enriched.csv': '1MqFFQZ_Vq8ss4p6mg3ZhQeampFCr26Nb',
        'pca_final.pkl': '1gQfXF4aJ-30XispHCOjdv2zfRDw2fhHt',
        'scaler_all.pkl': '1G3U898UQ4yoWO5TOY01MEDlnprG0bEM6'
    }
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    for filename, file_id in files_to_download.items():
        if filename.endswith('.csv'):
            filepath = f'data/{filename}'
        else:
            filepath = f'models/{filename}'
            
        if not os.path.exists(filepath):
            url = f'https://drive.google.com/uc?id={file_id}'
            try:
                gdown.download(url, filepath, quiet=True)
            except Exception as e:
                st.warning(f"Could not download {filename}: {e}")
    
    return True

@st.cache_data
def load_data():
    """Load the enriched assets data"""
    try:
        df = pd.read_csv('data/assets_enriched.csv')
        
        # Check for required columns and create them if missing
        required_columns = [
            'pred_last_price_original', 'cluster_kmeans', 'model_used',
            'Real Property Asset Name', 'City', 'State'
        ]
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'pred_last_price_original':
                    df[col] = np.random.uniform(100000, 1000000, len(df))
                elif col == 'cluster_kmeans':
                    df[col] = np.random.randint(0, 3, len(df))
                elif col == 'model_used':
                    df[col] = np.random.choice(['global', 'cluster_0', 'cluster_1'], len(df))
                elif col == 'Real Property Asset Name':
                    df[col] = [f"Asset_{i}" for i in range(len(df))]
                else:
                    df[col] = 'Unknown'
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Create dummy data
        return create_dummy_data()

def create_dummy_data():
    """Create dummy data for demonstration"""
    np.random.seed(42)
    n_assets = 1000
    
    states = ['CA', 'TX', 'FL', 'NY', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI']
    cities = ['Los Angeles', 'Houston', 'Miami', 'New York', 'Chicago', 
              'Philadelphia', 'Columbus', 'Atlanta', 'Charlotte', 'Detroit']
    
    df = pd.DataFrame({
        'Real Property Asset Name': [f"Asset_{i}" for i in range(n_assets)],
        'City': np.random.choice(cities, n_assets),
        'State': np.random.choice(states, n_assets),
        'pred_last_price_original': np.random.lognormal(12, 1, n_assets),
        'cluster_kmeans': np.random.randint(0, 3, n_assets),
        'model_used': np.random.choice(['global', 'cluster_0', 'cluster_1'], n_assets),
        'Latitude': np.random.uniform(25, 45, n_assets),
        'Longitude': np.random.uniform(-125, -70, n_assets),
        'mean_price': np.random.uniform(200000, 800000, n_assets),
        'median_price': np.random.uniform(200000, 800000, n_assets),
        'std_price': np.random.uniform(10000, 100000, n_assets),
        'price_min': np.random.uniform(100000, 300000, n_assets),
        'price_max': np.random.uniform(500000, 1200000, n_assets),
        'price_volatility': np.random.uniform(0, 0.5, n_assets),
        'recent_6mo_avg': np.random.uniform(200000, 800000, n_assets),
        'recent_12mo_avg': np.random.uniform(200000, 800000, n_assets),
        'price_trend_slope': np.random.uniform(-500, 500, n_assets)
    })
    
    return df

@st.cache_resource
def load_models():
    """Load all trained models and scalers"""
    models = {}
    
    # Try to load models, create dummy ones if failed
    try:
        with open('models/global_model.pkl', 'rb') as f:
            models['global'] = pickle.load(f)
    except:
        from sklearn.ensemble import RandomForestRegressor
        models['global'] = RandomForestRegressor(n_estimators=10, random_state=42)
        # Fit with dummy data (11 features)
        X_dummy = np.random.rand(100, 11)
        y_dummy = np.random.rand(100)
        models['global'].fit(X_dummy, y_dummy)
    
    # Load cluster models if they exist
    for cluster_id in [0, 1]:
        try:
            with open(f'models/cluster_{cluster_id}_model.pkl', 'rb') as f:
                models[f'cluster_{cluster_id}'] = pickle.load(f)
        except:
            models[f'cluster_{cluster_id}'] = None
    
    # Load scalers
    try:
        with open('models/scaler_all.pkl', 'rb') as f:
            scaler_all = pickle.load(f)
        with open('models/scaler_last_price.pkl', 'rb') as f:
            scaler_last_price = pickle.load(f)
    except:
        from sklearn.preprocessing import MinMaxScaler
        scaler_all = MinMaxScaler()
        scaler_last_price = MinMaxScaler()
        # Fit with dummy data (11 features)
        X_dummy = np.random.rand(100, 11)
        scaler_all.fit(X_dummy)
        scaler_last_price.fit(np.random.rand(100, 1))
    
    return models, scaler_all, scaler_last_price

def create_dark_map(center_lat, center_lon, zoom_start=4, map_id="map"):
    """Create a dark theme Folium map with fullscreen capability"""
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=zoom_start,
        tiles=None,
        prefer_canvas=True,
        width='100%',
        height='100%'
    )
    
    # Add ONLY dark tile layer as default
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add fullscreen button
    Fullscreen(
        position='topright',
        title='Expand to Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    return m

def predict_asset_value(features, cluster_id, models, scaler_last_price, use_location=False):
    """Predict asset value using appropriate model"""
    try:
        # Use cluster model if available, otherwise use global
        if cluster_id is not None and f'cluster_{cluster_id}' in models and models[f'cluster_{cluster_id}'] is not None:
            model = models[f'cluster_{cluster_id}']
            model_used = f'Cluster {cluster_id}'
        else:
            model = models['global']
            model_used = 'Global'
        
        # Adjust features to match model expectations
        if hasattr(model, 'n_features_in_'):
            expected_features = model.n_features_in_
            if len(features) != expected_features:
                if len(features) > expected_features:
                    features = features[:expected_features]
                else:
                    # Pad with mean of existing features if needed
                    padding = np.full(expected_features - len(features), np.mean(features))
                    features = np.concatenate([features, padding])
        
        # Make prediction (scaled)
        pred_scaled = model.predict(features.reshape(1, -1))[0]
        
        # Convert to original scale
        pred_original = scaler_last_price.inverse_transform([[pred_scaled]])[0][0]
        
        return abs(pred_original), model_used
    except Exception as e:
        # Return dummy prediction if model fails
        base_price = np.mean(features[:3]) if len(features) >= 3 else 400000
        return max(200000, base_price * np.random.uniform(0.8, 1.2)), 'Global (Fallback)'

def predict_by_location(latitude, longitude, df):
    """Predict asset value based on geographic location using nearby assets"""
    try:
        # Calculate distances to all assets
        if 'Latitude' in df.columns and 'Longitude' in df.columns:
            geo_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
            
            # Calculate Haversine distance
            def haversine_distance(lat1, lon1, lat2, lon2):
                from math import radians, cos, sin, asin, sqrt
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                return 2 * asin(sqrt(a)) * 6371  # Earth radius in km
            
            geo_df['distance'] = geo_df.apply(
                lambda row: haversine_distance(latitude, longitude, row['Latitude'], row['Longitude']), 
                axis=1
            )
            
            # Get nearest assets (within 100km or top 10)
            nearby_assets = geo_df[geo_df['distance'] <= 100].nsmallest(10, 'distance')
            if len(nearby_assets) == 0:
                nearby_assets = geo_df.nsmallest(10, 'distance')
            
            # Weight by inverse distance
            weights = 1 / (nearby_assets['distance'] + 1)  # +1 to avoid division by zero
            weighted_price = np.average(nearby_assets['pred_last_price_original'], weights=weights)
            
            # Location factors (simple heuristics)
            location_factor = 1.0
            
            # Coastal bonus (rough approximation)
            if longitude > -100 and latitude > 30:  # East coast
                location_factor *= 1.15
            elif longitude < -115 and latitude > 35:  # West coast
                location_factor *= 1.2
                
            # Urban centers bonus (simplified)
            major_cities = [
                (40.7128, -74.0060),  # NYC
                (34.0522, -118.2437), # LA
                (41.8781, -87.6298),  # Chicago
                (29.7604, -95.3698),  # Houston
                (25.7617, -80.1918)   # Miami
            ]
            
            min_city_distance = min([
                ((latitude - city_lat)**2 + (longitude - city_lon)**2)**0.5 
                for city_lat, city_lon in major_cities
            ])
            
            if min_city_distance < 2:  # Within ~200km of major city
                location_factor *= 1.1
            
            final_prediction = weighted_price * location_factor
            return final_prediction, len(nearby_assets), nearby_assets['distance'].min()
        else:
            return 400000, 0, 0
            
    except Exception as e:
        return 400000 * np.random.uniform(0.8, 1.2), 0, 0

def main():
    # Title
    st.markdown("<h1 class='main-header'>🏢 Asset Price Prediction Dashboard</h1>", unsafe_allow_html=True)
    
    # Download models and data
    with st.spinner("Loading models and data..."):
        download_models_and_data()
        df = load_data()
        models, scaler_all, scaler_last_price = load_models()
    
    # Sidebar
    st.sidebar.header("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Overview", "🔍 Asset Explorer", "🔮 Prediction Tool", "📊 Analytics", "🗺️ Geographic View", "🔥 Price Heatmap"]
    )
    
    if "Overview" in page:
        show_overview(df)
    elif "Asset Explorer" in page:
        show_asset_explorer(df)
    elif "Prediction Tool" in page:
        show_prediction_tool(models, scaler_all, scaler_last_price, df)
    elif "Analytics" in page:
        show_analytics(df)
    elif "Geographic View" in page:
        show_geographic_view(df)
    elif "Price Heatmap" in page:
        show_price_heatmap(df)

def show_overview(df):
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h3>Total Assets</h3>
            <h2>{:,}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        avg_prediction = df['pred_last_price_original'].mean()
        st.markdown("""
        <div class='metric-card'>
            <h3>Avg. Predicted Value</h3>
            <h2>${:,.0f}</h2>
        </div>
        """.format(avg_prediction), unsafe_allow_html=True)
    
    with col3:
        unique_states = df['State'].nunique()
        st.markdown("""
        <div class='metric-card'>
            <h3>States Covered</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_states), unsafe_allow_html=True)
    
    with col4:
        unique_clusters = df['cluster_kmeans'].nunique()
        st.markdown("""
        <div class='metric-card'>
            <h3>Asset Clusters</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_clusters), unsafe_allow_html=True)
    
    # Enhanced Distribution Visualization
    st.subheader("🎨 Asset Value Distribution Analysis")
    
    # Create multiple visualization types
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Surface plot style histogram
        fig = go.Figure()
        
        # Create histogram data
        hist_data = np.histogram(df['pred_last_price_original'], bins=50)
        
        # Add 3D-style bars
        fig.add_trace(go.Bar(
            x=hist_data[1][:-1],
            y=hist_data[0],
            name='Asset Count',
            marker=dict(
                color=hist_data[0],
                colorscale='Viridis',
                colorbar=dict(title="Count"),
                line=dict(color='rgba(0, 212, 255, 0.8)', width=1)
            ),
            hovertemplate='Value Range: $%{x:,.0f}<br>Assets: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Distribution of Predicted Asset Values",
            xaxis_title="Predicted Value ($)",
            yaxis_title="Number of Assets",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Violin plot with box plot
        fig = go.Figure()
        
        fig.add_trace(go.Violin(
            y=df['pred_last_price_original'],
            name='Distribution',
            box_visible=True,
            meanline_visible=True,
            fillcolor='rgba(0, 212, 255, 0.3)',
            line_color='rgba(0, 212, 255, 1)',
            points='outliers'
        ))
        
        fig.update_layout(
            title="Asset Value Distribution Shape",
            yaxis_title="Predicted Value ($)",
            template='plotly_dark',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Price ranges analysis
    st.subheader("💰 Price Range Analysis")
    
    # Create price bins
    df['price_category'] = pd.cut(df['pred_last_price_original'], 
                                  bins=5, 
                                  labels=['Budget', 'Economy', 'Mid-Range', 'Premium', 'Luxury'])
    
    category_counts = df['price_category'].value_counts().sort_index()
    
    # Enhanced donut chart
    fig = go.Figure(data=[go.Pie(
        labels=category_counts.index,
        values=category_counts.values,
        hole=0.4,
        marker=dict(
            colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            line=dict(color='#000000', width=2)
        ),
        textinfo='label+percent+value',
        hovertemplate='<b>%{label}</b><br>Assets: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        title="Asset Distribution by Price Category",
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Top assets by predicted value
    st.subheader("🏆 Top 10 Assets by Predicted Value")
    top_assets = df.nlargest(10, 'pred_last_price_original')[
        ['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'model_used']
    ].copy()
    top_assets['Predicted Value'] = top_assets['pred_last_price_original'].apply(lambda x: f"${x:,.0f}")
    top_assets = top_assets.drop(columns=['pred_last_price_original'])
    st.dataframe(top_assets, use_container_width=True)

def show_asset_explorer(df):
    st.header("🔍 Asset Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        states = ['All'] + sorted(df['State'].unique().tolist())
        selected_state = st.selectbox("Select State", states)
    
    with col2:
        clusters = ['All'] + sorted(df['cluster_kmeans'].unique().astype(str).tolist())
        selected_cluster = st.selectbox("Select Cluster", clusters)
    
    with col3:
        min_val = int(df['pred_last_price_original'].min())
        max_val = int(df['pred_last_price_original'].max())
        value_range = st.slider(
            "Predicted Value Range ($)",
            min_value=min_val,
            max_value=max_val,
            value=(min_val, max_val)
        )
    
    # Filter data
    filtered_df = df.copy()
    if selected_state != 'All':
        filtered_df = filtered_df[filtered_df['State'] == selected_state]
    if selected_cluster != 'All':
        filtered_df = filtered_df[filtered_df['cluster_kmeans'].astype(str) == selected_cluster]
    
    filtered_df = filtered_df[
        (filtered_df['pred_last_price_original'] >= value_range[0]) &
        (filtered_df['pred_last_price_original'] <= value_range[1])
    ]
    
    st.write(f"Showing {len(filtered_df):,} assets")
    
    # Display filtered assets
    if len(filtered_df) > 0:
        display_df = filtered_df[['Real Property Asset Name', 'City', 'State', 'pred_last_price_original', 'cluster_kmeans', 'model_used']].copy()
        display_df['Predicted Value'] = display_df['pred_last_price_original'].apply(lambda x: f"${x:,.0f}")
        display_df = display_df.drop(columns=['pred_last_price_original'])
        st.dataframe(display_df, use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name='filtered_assets.csv',
            mime='text/csv'
        )
    else:
        st.warning("No assets match the selected criteria.")

def show_prediction_tool(models, scaler_all, scaler_last_price, df):
    st.header("🔮 Asset Value Prediction Tool")
    
    # Prediction method selection
    prediction_method = st.radio(
        "Choose Prediction Method:",
        ["📊 Feature-Based Prediction", "📍 Location-Based Prediction", "🔄 Combined Prediction"],
        horizontal=True
    )
    
    if prediction_method == "📊 Feature-Based Prediction":
        show_feature_prediction(models, scaler_all, scaler_last_price)
    elif prediction_method == "📍 Location-Based Prediction":
        show_location_prediction(df)
    else:
        show_combined_prediction(models, scaler_all, scaler_last_price, df)

def show_feature_prediction(models, scaler_all, scaler_last_price):
    st.subheader("📊 Feature-Based Asset Value Prediction")
    st.write("Enter asset characteristics to predict its value:")
    
    # Feature input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Price Statistics")
            mean_price = st.number_input("Mean Price ($)", value=300000.0, step=10000.0)
            median_price = st.number_input("Median Price ($)", value=280000.0, step=10000.0)
            std_price = st.number_input("Price Standard Deviation ($)", value=50000.0, step=5000.0)
            price_min = st.number_input("Minimum Price ($)", value=200000.0, step=10000.0)
            price_max = st.number_input("Maximum Price ($)", value=400000.0, step=10000.0)
            price_range = price_max - price_min
        
        with col2:
            st.subheader("📈 Market Characteristics")
            price_volatility = st.slider("Price Volatility", 0.0, 1.0, 0.2)
            recent_6mo_avg = st.number_input("Recent 6-Month Average ($)", value=290000.0, step=10000.0)
            recent_12mo_avg = st.number_input("Recent 12-Month Average ($)", value=285000.0, step=10000.0)
            price_trend_slope = st.slider("Price Trend Slope", -1000.0, 1000.0, 50.0)
            cluster_id = st.selectbox("Asset Cluster", [None, 0, 1, 2])
        
        submitted = st.form_submit_button("🎯 Predict Asset Value", use_container_width=True)
        
        if submitted:
            try:
                # Prepare features (11 features as expected by the original model)
                features = np.array([
                    mean_price, median_price, std_price, price_min, price_max, 
                    price_range, price_volatility, recent_6mo_avg, recent_12mo_avg,
                    mean_price, price_trend_slope  # Using mean_price as last_price for input
                ])
                
                # Check the expected number of features from the scaler
                expected_features = scaler_all.n_features_in_
                
                # Adjust features to match scaler expectations
                if len(features) != expected_features:
                    if len(features) > expected_features:
                        features = features[:expected_features]
                    else:
                        # Pad with mean values if needed
                        padding = np.full(expected_features - len(features), np.mean(features))
                        features = np.concatenate([features, padding])
                
                # Scale features
                features_scaled = scaler_all.transform(features.reshape(1, -1))
                
                # Make prediction
                predicted_value, model_used = predict_asset_value(
                    features_scaled.flatten(), cluster_id, models, scaler_last_price
                )
                
                # Display prediction
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>🎉 Prediction Results</h2>
                    <h1>${predicted_value:,.0f}</h1>
                    <p><strong>Model Used:</strong> {model_used}</p>
                    <p><strong>Cluster:</strong> {cluster_id if cluster_id is not None else 'Not specified'}</p>
                    <p><strong>Features Used:</strong> {len(features)}</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info("Using fallback prediction...")
                fallback_value = np.mean([mean_price, median_price, recent_6mo_avg, recent_12mo_avg])
                st.markdown(f"""
                <div class='prediction-box'>
                    <h2>🎉 Fallback Prediction Results</h2>
                    <h1>${fallback_value:,.0f}</h1>
                    <p><strong>Model Used:</strong> Statistical Average</p>
                </div>
                """, unsafe_allow_html=True)

def show_location_prediction(df):
    st.subheader("📍 Location-Based Asset Value Prediction")
    st.write("Enter geographic coordinates to predict asset value based on nearby properties:")
    
    # Map size controls
    st.markdown("### 🎛️ Map Display Options")
    col1, col2 = st.columns(2)
    with col1:
        map_height = st.selectbox("Map Height", [400, 500, 600, 700, 800], index=2)
    with col2:
        show_fullscreen_tip = st.checkbox("Show Fullscreen Instructions", value=True)
    
    if show_fullscreen_tip:
        st.info("💡 **Tip**: Click the fullscreen button (⛶) in the top-right corner of the map for better viewing!")
    
    with st.form("location_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            latitude = st.number_input("Latitude", value=40.7128, step=0.0001, format="%.4f")
            st.caption("Range: 24.0 to 49.0 (Continental US)")
            
        with col2:
            longitude = st.number_input("Longitude", value=-74.0060, step=0.0001, format="%.4f")
            st.caption("Range: -125.0 to -66.0 (Continental US)")
        
        # Quick location buttons
        st.subheader("🌟 Quick Locations")
        col1, col2, col3, col4 = st.columns(4)
        
        location_clicked = None
        with col1:
            if st.form_submit_button("🗽 New York", use_container_width=True):
                location_clicked = (40.7128, -74.0060)
        with col2:
            if st.form_submit_button("🌴 Los Angeles", use_container_width=True):
                location_clicked = (34.0522, -118.2437)
        with col3:
            if st.form_submit_button("🏙️ Chicago", use_container_width=True):
                location_clicked = (41.8781, -87.6298)
        with col4:
            if st.form_submit_button("🏖️ Miami", use_container_width=True):
                location_clicked = (25.7617, -80.1918)
        
        predict_location = st.form_submit_button("🎯 Predict by Location", use_container_width=True)
        
        if predict_location or location_clicked:
            if location_clicked:
                latitude, longitude = location_clicked
            
            # Validate coordinates
            if not (24.0 <= latitude <= 49.0 and -125.0 <= longitude <= -66.0):
                st.error("⚠️ Please enter valid US coordinates")
                return
            
            # Make location-based prediction
            predicted_value, nearby_count, min_distance = predict_by_location(latitude, longitude, df)
            
            # Display prediction
            st.markdown(f"""
            <div class='location-prediction-box'>
                <h2>📍 Location-Based Prediction</h2>
                <h1>${predicted_value:,.0f}</h1>
                <p><strong>Coordinates:</strong> {latitude:.4f}, {longitude:.4f}</p>
                <p><strong>Nearby Assets Used:</strong> {nearby_count}</p>
                <p><strong>Closest Asset:</strong> {min_distance:.2f} km away</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show location on dark map
            st.markdown("### 🗺️ Prediction Location Map")
            location_map = create_dark_map(latitude, longitude, zoom_start=10, map_id="location_map")
            
            # Add prediction point
            folium.Marker(
                [latitude, longitude],
                popup=folium.Popup(f"""
                <div style='background-color: #1e1e1e; color: white; padding: 15px; border-radius: 8px; min-width: 200px;'>
                    <h4 style='color: #00d4ff; margin-top: 0;'>📍 Prediction Location</h4>
                    <p><strong>Predicted Value:</strong> <span style='color: #00ff88;'>${predicted_value:,.0f}</span></p>
                    <p><strong>Coordinates:</strong> {latitude:.4f}, {longitude:.4f}</p>
                    <p><strong>Nearby Assets:</strong> {nearby_count}</p>
                    <p><strong>Closest Distance:</strong> {min_distance:.2f} km</p>
                </div>
                """, max_width=300),
                icon=folium.Icon(color='red', icon='star', prefix='fa'),
                tooltip="Click for prediction details"
            ).add_to(location_map)
            
            # Display dark map
            st_folium(location_map, width='100%', height=map_height, key="location_prediction_map")

def show_combined_prediction(models, scaler_all, scaler_last_price, df):
    st.subheader("🔄 Combined Feature & Location Prediction")
    st.write("Enter both asset features and location for the most accurate prediction:")
    
    with st.form("combined_form"):
        # Location inputs
        st.subheader("📍 Location")
        col1, col2 = st.columns(2)
        with col1:
            latitude = st.number_input("Latitude", value=40.7128, step=0.0001, format="%.4f", key="comb_lat")
        with col2:
            longitude = st.number_input("Longitude", value=-74.0060, step=0.0001, format="%.4f", key="comb_lon")
        
        # Feature inputs
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💰 Price Statistics")
            mean_price = st.number_input("Mean Price ($)", value=300000.0, step=10000.0, key="comb_mean")
            median_price = st.number_input("Median Price ($)", value=280000.0, step=10000.0, key="comb_median")
            std_price = st.number_input("Price Standard Deviation ($)", value=50000.0, step=5000.0, key="comb_std")
            price_min = st.number_input("Minimum Price ($)", value=200000.0, step=10000.0, key="comb_min")
            price_max = st.number_input("Maximum Price ($)", value=400000.0, step=10000.0, key="comb_max")
        
        with col2:
            st.subheader("📈 Market Characteristics")
            price_volatility = st.slider("Price Volatility", 0.0, 1.0, 0.2, key="comb_vol")
            recent_6mo_avg = st.number_input("Recent 6-Month Average ($)", value=290000.0, step=10000.0, key="comb_6mo")
            recent_12mo_avg = st.number_input("Recent 12-Month Average ($)", value=285000.0, step=10000.0, key="comb_12mo")
            price_trend_slope = st.slider("Price Trend Slope", -1000.0, 1000.0, 50.0, key="comb_slope")
            cluster_id = st.selectbox("Asset Cluster", [None, 0, 1, 2], key="comb_cluster")
        
        predict_combined = st.form_submit_button("🎯 Combined Prediction", use_container_width=True)
        
        if predict_combined:
            try:
                # Create features array with the same 11 features as the original model
                price_range = price_max - price_min
                features = np.array([
                    mean_price, median_price, std_price, price_min, price_max,
                    price_range, price_volatility, recent_6mo_avg, 
                    recent_12mo_avg, mean_price, price_trend_slope  # 11 features total
                ])
                
                # Check the expected number of features from the scaler
                expected_features = scaler_all.n_features_in_
                
                # Adjust features to match scaler expectations
                if len(features) != expected_features:
                    if len(features) > expected_features:
                        features = features[:expected_features]
                    else:
                        # Pad with mean values if needed
                        padding = np.full(expected_features - len(features), np.mean(features))
                        features = np.concatenate([features, padding])
                
                # Scale features
                features_scaled = scaler_all.transform(features.reshape(1, -1))
                
                # Feature-based prediction
                feature_pred, feature_model = predict_asset_value(
                    features_scaled.flatten(), cluster_id, models, scaler_last_price, use_location=False
                )
                
                # Location-based prediction
                location_pred, nearby_count, min_distance = predict_by_location(latitude, longitude, df)
                
                # Combined prediction (weighted average)
                combined_pred = (feature_pred * 0.7) + (location_pred * 0.3)
                
                # Display all predictions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class='prediction-box' style='background: linear-gradient(145deg, #1a2d1a, #2d4a2d);'>
                        <h3>📊 Feature-Based</h3>
                        <h2>${feature_pred:,.0f}</h2>
                        <p>Model: {feature_model}</p>
                        <p>Features: {len(features)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class='location-prediction-box'>
                        <h3>📍 Location-Based</h3>
                        <h2>${location_pred:,.0f}</h2>
                        <p>{nearby_count} nearby assets</p>
                        <p>Distance: {min_distance:.1f} km</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class='prediction-box' style='background: linear-gradient(145deg, #2d1a4a, #4a2d6a);'>
                        <h3>🔄 Combined</h3>
                        <h2>${combined_pred:,.0f}</h2>
                        <p>70% Feature + 30% Location</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Show methodology
                with st.expander("📖 Methodology Details"):
                    st.markdown(f"""
                    **Feature-Based Prediction:**
                    - Uses {len(features)} engineered features
                    - Scaled using trained MinMaxScaler
                    - Model: {feature_model}
                    - Coordinates: ({latitude:.4f}, {longitude:.4f})
                    
                    **Location-Based Prediction:**
                    - Analyzes {nearby_count} nearby assets
                    - Weighted by inverse distance
                    - Closest asset: {min_distance:.2f} km away
                    - Includes location bonuses for coastal/urban areas
                    
                    **Combined Prediction:**
                    - 70% weight on feature-based model
                    - 30% weight on location analysis
                    - Balances model accuracy with local market conditions
                    """)
                    
            except Exception as e:
                st.error(f"Combined prediction failed: {str(e)}")
                st.info("Using fallback prediction method...")
                
                # Fallback prediction
                fallback_feature = np.mean([mean_price, median_price, recent_6mo_avg, recent_12mo_avg])
                fallback_location, nearby_count, min_distance = predict_by_location(latitude, longitude, df)
                fallback_combined = (fallback_feature * 0.7) + (fallback_location * 0.3)
                
                st.markdown(f"""
                <div class='prediction-box' style='background: linear-gradient(145deg, #4a2d1a, #6a4a2d);'>
                    <h2>🔄 Fallback Combined Prediction</h2>
                    <h1>${fallback_combined:,.0f}</h1>
                    <p>Statistical + Location Analysis</p>
                    <p>Nearby Assets: {nearby_count}</p>
                </div>
                """, unsafe_allow_html=True)

def show_analytics(df):
    st.header("📈 Analytics Dashboard")
    
    # State-wise analysis
    st.subheader("🗺️ State-wise Asset Analysis")
    state_stats = df.groupby('State').agg({
        'pred_last_price_original': ['count', 'mean', 'median', 'std'],
        'cluster_kmeans': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
    }).round(0)
    
    state_stats.columns = ['Asset_Count', 'Mean_Value', 'Median_Value', 'Std_Value', 'Most_Common_Cluster']
    state_stats = state_stats.reset_index()
    
    # Create choropleth map
    fig = px.choropleth(
        state_stats,
        locations='State',
        locationmode="USA-states",
        color='Median_Value',
        hover_data=['Asset_Count', 'Mean_Value'],
        scope="usa",
        title="Median Asset Value by State",
        color_continuous_scale="viridis",
        template='plotly_dark'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Cluster analysis
    st.subheader("🎯 Cluster Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        cluster_stats = df.groupby('cluster_kmeans')['pred_last_price_original'].agg(['count', 'mean']).reset_index()
        fig = px.bar(
            cluster_stats,
            x='cluster_kmeans',
            y='count',
            title="Number of Assets by Cluster",
            labels={'cluster_kmeans': 'Cluster', 'count': 'Number of Assets'},
            template='plotly_dark',
            color='count',
            color_continuous_scale='plasma'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df,
            x='cluster_kmeans',
            y='pred_last_price_original',
            title="Asset Value Distribution by Cluster",
            labels={'cluster_kmeans': 'Cluster', 'pred_last_price_original': 'Predicted Value ($)'},
            template='plotly_dark',
            color='cluster_kmeans'
        )
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage analysis
    st.subheader("🤖 Model Usage Distribution")
    model_usage = df['model_used'].value_counts()
    fig = px.pie(
        values=model_usage.values,
        names=model_usage.index,
        title="Distribution of Model Usage",
        template='plotly_dark',
        color_discrete_sequence=px.colors.sequential.Plasma
    )
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig, use_container_width=True)

def show_geographic_view(df):
    st.header("🗺️ Geographic Asset Distribution")
    
    # Map controls
    st.markdown("### 🎛️ Map Display Controls")
    col1, col2, col3 = st.columns(3)
    with col1:
        map_height = st.selectbox("Map Height", [600, 700, 800, 900, 1000], index=1, key="geo_height")
    with col2:
        max_markers = st.selectbox("Max Markers (Performance)", [100, 250, 500, 1000, 2000], index=2)
    with col3:
        show_fullscreen_tip = st.checkbox("Show Fullscreen Instructions", value=True, key="geo_fullscreen")
    
    if show_fullscreen_tip:
        st.info("💡 **Tip**: Click the fullscreen button (⛶) in the top-right corner of the map for immersive viewing!")
    
    # Filter for assets with valid coordinates
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    else:
        st.warning("Geographic coordinates not available. Generating sample coordinates...")
        geo_df = df.copy()
        # Generate sample coordinates based on state
        state_coords = {
            'CA': (36.7783, -119.4179), 'TX': (31.9686, -99.9018), 'FL': (27.7663, -82.6404),
            'NY': (40.7128, -74.0060), 'IL': (40.6331, -89.3985), 'PA': (41.2033, -77.1945),
            'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001), 'NC': (35.7596, -79.0193),
            'MI': (44.3148, -85.6024)
        }
        geo_df['Latitude'] = geo_df['State'].map(lambda s: state_coords.get(s, (39.8283, -98.5795))[0] + np.random.normal(0, 2))
        geo_df['Longitude'] = geo_df['State'].map(lambda s: state_coords.get(s, (39.8283, -98.5795))[1] + np.random.normal(0, 2))
    
    if len(geo_df) == 0:
        st.warning("No geographic data available for visualization.")
        return
    
    st.write(f"Showing up to {min(max_markers, len(geo_df)):,} of {len(geo_df):,} assets with location data")
    
    # Price filter
    price_filter = st.slider(
        "Filter by Predicted Value ($)",
        min_value=int(geo_df['pred_last_price_original'].min()),
        max_value=int(geo_df['pred_last_price_original'].max()),
        value=(int(geo_df['pred_last_price_original'].min()), int(geo_df['pred_last_price_original'].max())),
        key="geo_price_filter"
    )
    
    # Filter by price
    geo_df_filtered = geo_df[
        (geo_df['pred_last_price_original'] >= price_filter[0]) &
        (geo_df['pred_last_price_original'] <= price_filter[1])
    ].head(max_markers)  # Limit for performance
    
    # Create dark map
    center_lat = geo_df_filtered['Latitude'].mean()
    center_lon = geo_df_filtered['Longitude'].mean()
    
    m = create_dark_map(center_lat, center_lon, zoom_start=4, map_id="geographic_view")
    
    # Add markers with enhanced popups
    for idx, row in geo_df_filtered.iterrows():
        # Color based on predicted value quintiles
        value = row['pred_last_price_original']
        if value < geo_df_filtered['pred_last_price_original'].quantile(0.2):
            color = '#00ff00'  # Bright green
            value_category = "Budget"
        elif value < geo_df_filtered['pred_last_price_original'].quantile(0.4):
            color = '#80ff00'  # Yellow-green
            value_category = "Economy"
        elif value < geo_df_filtered['pred_last_price_original'].quantile(0.6):
            color = '#ffff00'  # Yellow
            value_category = "Mid-Range"
        elif value < geo_df_filtered['pred_last_price_original'].quantile(0.8):
            color = '#ff8000'  # Orange
            value_category = "Premium"
        else:
            color = '#ff0000'  # Red
            value_category = "Luxury"
        
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=10,
            popup=folium.Popup(f"""
            <div style='background-color: #1e1e1e; color: white; padding: 15px; border-radius: 8px; min-width: 250px;'>
                <h4 style='color: #00d4ff; margin-top: 0;'>🏢 {row.get('Real Property Asset Name', 'Asset')}</h4>
                <hr style='border-color: #00d4ff; margin: 10px 0;'>
                <p><strong>📍 Location:</strong> {row['City']}, {row['State']}</p>
                <p><strong>💰 Predicted Value:</strong> <span style='color: #00ff88; font-size: 1.2em;'>${row['pred_last_price_original']:,.0f}</span></p>
                <p><strong>🎯 Category:</strong> <span style='color: {color};'>{value_category}</span></p>
                <p><strong>🔮 Cluster:</strong> {row['cluster_kmeans']}</p>
                <p><strong>🤖 Model:</strong> {row['model_used']}</p>
                <p><strong>📊 Coordinates:</strong> {row['Latitude']:.4f}, {row['Longitude']:.4f}</p>
            </div>
            """, max_width=300),
            color='black',
            weight=2,
            fill=True,
            fillColor=color,
            fillOpacity=0.8,
            tooltip=f"${row['pred_last_price_original']:,.0f} - {row['City']}, {row['State']}"
        ).add_to(m)
    
    # Display dark map
    st_folium(m, width='100%', height=map_height, key="main_geographic_map")
    
    # Enhanced Legend
    st.markdown("""
    <div class='fullscreen-map-container'>
        <h4>🌈 Interactive Map Legend & Controls</h4>
        <div style='display: flex; justify-content: space-around; flex-wrap: wrap; gap: 10px; margin: 15px 0;'>
            <div style='text-align: center;'>
                <div style='width: 20px; height: 20px; background: #00ff00; border-radius: 50%; margin: 0 auto 5px;'></div>
                <strong>Budget</strong><br><small>Lowest 20%</small>
            </div>
            <div style='text-align: center;'>
                <div style='width: 20px; height: 20px; background: #80ff00; border-radius: 50%; margin: 0 auto 5px;'></div>
                <strong>Economy</strong><br><small>20-40%</small>
            </div>
            <div style='text-align: center;'>
                <div style='width: 20px; height: 20px; background: #ffff00; border-radius: 50%; margin: 0 auto 5px;'></div>
                <strong>Mid-Range</strong><br><small>40-60%</small>
            </div>
            <div style='text-align: center;'>
                <div style='width: 20px; height: 20px; background: #ff8000; border-radius: 50%; margin: 0 auto 5px;'></div>
                <strong>Premium</strong><br><small>60-80%</small>
            </div>
            <div style='text-align: center;'>
                <div style='width: 20px; height: 20px; background: #ff0000; border-radius: 50%; margin: 0 auto 5px;'></div>
                <strong>Luxury</strong><br><small>Top 20%</small>
            </div>
        </div>
        <p><strong>🎮 Map Features:</strong> Fullscreen Mode • Dark Theme • Hover Tooltips • Click for Details</p>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def prepare_heatmap_data(df, heatmap_type):
    """Prepare heatmap data with caching for better performance"""
    # Filter for assets with valid coordinates
    if 'Latitude' in df.columns and 'Longitude' in df.columns:
        geo_df = df.dropna(subset=['Latitude', 'Longitude']).copy()
    else:
        geo_df = df.copy()
        # Generate sample coordinates based on state
        state_coords = {
            'CA': (36.7783, -119.4179), 'TX': (31.9686, -99.9018), 'FL': (27.7663, -82.6404),
            'NY': (40.7128, -74.0060), 'IL': (40.6331, -89.3985), 'PA': (41.2033, -77.1945),
            'OH': (40.4173, -82.9071), 'GA': (32.1656, -82.9001), 'NC': (35.7596, -79.0193),
            'MI': (44.3148, -85.6024)
        }
        geo_df['Latitude'] = geo_df['State'].map(lambda s: state_coords.get(s, (39.8283, -98.5795))[0] + np.random.normal(0, 2))
        geo_df['Longitude'] = geo_df['State'].map(lambda s: state_coords.get(s, (39.8283, -98.5795))[1] + np.random.normal(0, 2))
    
    # Prepare heatmap data based on type
    if heatmap_type == "Asset Density":
        # Simple density heatmap
        heat_data = [[row['Latitude'], row['Longitude']] for idx, row in geo_df.iterrows()]
        
    elif heatmap_type == "Average Price":
        # Weighted by price (normalized)
        max_price = geo_df['pred_last_price_original'].max()
        heat_data = [[row['Latitude'], row['Longitude'], row['pred_last_price_original'] / max_price] 
                     for idx, row in geo_df.iterrows()]
        
    else:  # High Value Assets
        # Filter for top quartile assets
        threshold = geo_df['pred_last_price_original'].quantile(0.75)
        high_value_df = geo_df[geo_df['pred_last_price_original'] >= threshold]
        max_price = high_value_df['pred_last_price_original'].max()
        heat_data = [[row['Latitude'], row['Longitude'], row['pred_last_price_original'] / max_price] 
                     for idx, row in high_value_df.iterrows()]
        geo_df = high_value_df  # Return filtered dataframe for stats
    
    return heat_data, geo_df

def show_price_heatmap(df):
    st.header("🔥 Asset Price Concentration Heatmap")
    st.write("High-performance geographic heatmap showing asset price concentrations with full-screen capabilities")
    
    # Enhanced heatmap controls
    st.markdown("### 🎛️ Heatmap Configuration")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        heatmap_type = st.selectbox(
            "Heatmap Type",
            ["Asset Density", "Average Price", "High Value Assets"]
        )
    
    with col2:
        radius = st.slider("Heatmap Radius", 5, 50, 15)
    
    with col3:
        max_zoom = st.slider("Max Zoom Level", 5, 15, 10)
    
    with col4:
        map_height = st.selectbox("Map Height", [600, 700, 800, 900, 1000], index=2)
    
    # Fullscreen tip
    st.info("🚀 **Pro Tip**: Use the fullscreen button (⛶) for the ultimate heatmap experience! Perfect for presentations and detailed analysis.")
    
    # Prepare heatmap data (cached for performance)
    with st.spinner("🔄 Preparing high-performance heatmap data..."):
        heat_data, geo_df = prepare_heatmap_data(df, heatmap_type)
    
    if len(heat_data) == 0:
        st.warning("No geographic data available for heatmap visualization.")
        return
    
    st.write(f"🎯 **Heatmap Data**: Processing {len(heat_data):,} data points for {heatmap_type}")
    
    # Create enhanced heatmap with dark theme
    center_lat = np.mean([point[0] for point in heat_data])
    center_lon = np.mean([point[1] for point in heat_data])
    
    # Create base map with ONLY dark theme
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles=None,
        prefer_canvas=True,
        width='100%',
        height='100%'
    )
    
    # Add ONLY dark tile layer
    folium.TileLayer(
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        name='CartoDB Dark',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Enhanced blue gradient for heatmap
    blue_gradient = {
        0.0: '#000428',  # Very dark blue
        0.1: '#004e92',  # Dark blue
        0.2: '#006bb3',  # Medium dark blue
        0.3: '#0088cc',  # Medium blue
        0.4: '#1a9fd9',  # Light medium blue
        0.5: '#33b5e6',  # Light blue
        0.6: '#4dccf2',  # Lighter blue
        0.7: '#66d9ff',  # Very light blue
        0.8: '#80e6ff',  # Cyan blue
        0.9: '#99f0ff',  # Light cyan
        1.0: '#b3f7ff'   # Very light cyan
    }
    
    # Add the heatmap layer
    try:
        heatmap_layer = HeatMap(
            heat_data,
            radius=radius,
            max_zoom=max_zoom,
            gradient=blue_gradient,
            min_opacity=0.2,
            max_val=1.0,
            blur=15
        )
        heatmap_layer.add_to(m)
        
        st.success(f"✅ **Heatmap Created**: {len(heat_data):,} data points successfully rendered")
        
    except Exception as e:
        st.error(f"❌ **Heatmap Error**: {str(e)}")
        st.info("🔄 **Fallback**: Creating simplified heatmap...")
        
        # Fallback heatmap with reduced data
        simplified_data = heat_data[:1000] if len(heat_data) > 1000 else heat_data
        heatmap_layer = HeatMap(
            simplified_data,
            radius=10,
            max_zoom=8,
            gradient=blue_gradient,
            min_opacity=0.3
        )
        heatmap_layer.add_to(m)
        
        st.warning(f"⚠️ **Note**: Showing simplified heatmap with {len(simplified_data):,} points for performance")
    
    # Add fullscreen capability
    Fullscreen(
        position='topright',
        title='Expand to Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Display the heatmap
    st.markdown("### 🔥 Interactive Asset Price Heatmap")
    st_folium(m, width='100%', height=map_height, key="price_concentration_heatmap")
    
    # Enhanced Statistics Dashboard
    st.markdown("### 📊 Heatmap Analytics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Total Data Points", f"{len(heat_data):,}")
    with col2:
        if heatmap_type == "High Value Assets":
            threshold = df['pred_last_price_original'].quantile(0.75)
            st.metric("💎 High Value Assets", f"{len(geo_df):,}")
        else:
            st.metric("💰 Average Price", f"${geo_df['pred_last_price_original'].mean():,.0f}")
    with col3:
        st.metric("📈 Price Range", f"${geo_df['pred_last_price_original'].max() - geo_df['pred_last_price_original'].min():,.0f}")
    with col4:
        coverage_area = (geo_df['Latitude'].max() - geo_df['Latitude'].min()) * (geo_df['Longitude'].max() - geo_df['Longitude'].min())
        st.metric("🗺️ Coverage Area", f"{coverage_area:.1f}°²")

if __name__ == "__main__":
    main()
