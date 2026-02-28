"""Streamlit application for diabetes prediction visualization and interaction"""
import sys
from pathlib import Path

# Add project root to path (go up one level from scripts/ to project root)
# Use resolve() to get absolute path to avoid path resolution issues
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient
from src.data.load_data import load_dataset
from src.data.preprocessing import engineer_features, encode_categorical_features, ScalerManager
from src.models.evaluate import get_best_model, get_model_comparison
from src.utils.config import load_config

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration - use project_root calculated at top of file
config = load_config(str(project_root / "configs" / "config.yaml"))

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
experiment_name = config['mlflow']['experiment_name']


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    # Try multiple possible paths in order of preference
    possible_paths = [
        # Primary: relative to project root from config
        project_root / config['data']['raw_data_path'],
        # Secondary: explicit path from project root
        project_root / "data" / "raw" / "diabetes_dataset.xlsx",
        # Tertiary: relative to current working directory
        Path.cwd() / config['data']['raw_data_path'],
        Path.cwd() / "data" / "raw" / "diabetes_dataset.xlsx",
        # Fallback: root directory (in case file wasn't moved)
        project_root / "diabetes_dataset.xlsx",
        Path.cwd() / "diabetes_dataset.xlsx",
    ]
    
    data_path = None
    tried_paths = []
    
    for path in possible_paths:
        # Resolve the path
        try:
            if path.is_absolute():
                resolved_path = path
            else:
                # Try relative to project root first, then current directory
                resolved_path = (project_root / path).resolve()
                if not resolved_path.exists():
                    resolved_path = (Path.cwd() / path).resolve()
        except Exception:
            resolved_path = path
        
        tried_paths.append(str(resolved_path))
        
        if resolved_path.exists() and resolved_path.is_file():
            data_path = resolved_path
            break
    
    # Final check with helpful error message
    if data_path is None or not data_path.exists():
        st.error("❌ **Data file not found!**")
        st.markdown("---")
        
        # Check if running on Streamlit Cloud
        is_streamlit_cloud = "/mount/src/" in str(project_root) or "/mount/src/" in str(Path.cwd())
        
        if is_streamlit_cloud:
            st.warning("🌐 **Running on Streamlit Cloud detected**")
            st.info("On Streamlit Cloud, ensure `diabetes_dataset.xlsx` is committed to your repository in the `data/raw/` directory.")
        
        st.info(f"**Project root:** `{project_root}`")
        st.info(f"**Current working directory:** `{Path.cwd()}`")
        st.info(f"**Expected path:** `{config['data']['raw_data_path']}`")
        st.markdown("---")
        st.warning("**Tried the following paths:**")
        for i, path_str in enumerate(tried_paths, 1):
            path_obj = Path(path_str)
            exists = "✅" if path_obj.exists() and path_obj.is_file() else "❌"
            st.text(f"{i}. {exists} {path_str}")
        st.markdown("---")
        st.error("**Solution:** Please ensure `diabetes_dataset.xlsx` is located at:")
        st.code(f"{project_root / 'data' / 'raw' / 'diabetes_dataset.xlsx'}", language=None)
        
        if not is_streamlit_cloud:
            st.info("💡 **Tip:** If the file is in the project root, move it to `data/raw/` directory.")
        else:
            st.info("💡 **For Streamlit Cloud:** Commit the file to your repository and redeploy.")
        
        st.stop()
    
    # Log successful path for debugging
    st.sidebar.success(f"✅ Data loaded from: `{data_path}`")
    
    df = load_dataset(str(data_path))
    return df


@st.cache_data
def get_processed_data(df):
    """Process the data"""
    df_processed = engineer_features(df)
    df_encoded = encode_categorical_features(df_processed)
    return df_encoded


@st.cache_data
def get_model_comparison_data():
    """Get model comparison from MLflow"""
    try:
        return get_model_comparison(experiment_name)
    except Exception as e:
        st.error(f"Error loading model comparison: {str(e)}")
        return None


@st.cache_data
def get_best_model_data():
    """Get best model information"""
    try:
        return get_best_model(experiment_name)
    except Exception as e:
        st.error(f"Error loading best model: {str(e)}")
        return None


def load_model_from_mlflow(run_id: str, model_name: str):
    """Load model from MLflow with error handling and fallback"""
    try:
        model_uri = f"runs:/{run_id}/model"
        
        if 'XGBoost' in model_name:
            return mlflow.xgboost.load_model(model_uri)
        else:
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        # If the specific run fails, try to get the latest run
        st.warning(f"⚠️ Could not load model from run {run_id}. Trying to find latest model...")
        
        try:
            client = MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            
            if experiment is None:
                raise ValueError(f"Experiment '{experiment_name}' not found")
            
            # Get all runs sorted by creation time (newest first)
            runs = client.search_runs(
                experiment.experiment_id,
                order_by=["start_time DESC"],
                max_results=1
            )
            
            if not runs:
                raise ValueError("No runs found in experiment")
            
            # Use the latest run
            latest_run = runs[0]
            latest_run_id = latest_run.info.run_id
            latest_model_name = latest_run.data.tags.get('mlflow.runName', latest_run_id)
            
            st.info(f"✅ Using latest model: {latest_model_name} (Run ID: {latest_run_id})")
            
            model_uri = f"runs:/{latest_run_id}/model"
            
            if 'XGBoost' in latest_model_name:
                return mlflow.xgboost.load_model(model_uri)
            else:
                return mlflow.sklearn.load_model(model_uri)
                
        except Exception as fallback_error:
            raise Exception(
                f"Failed to load model from run {run_id}. "
                f"Fallback also failed: {str(fallback_error)}. "
                f"Please run training first: `python scripts/train.py`"
            )


def predict_diabetes(weight: float, height: float, hair_color: str, model, scaler, feature_columns):
    """Make prediction for a single sample"""
    # Create feature vector
    features = {
        'Peso': weight,
        'Altura': height,
        'BMI': weight / ((height / 100) ** 2)
    }
    
    # Add hair color features
    hair_colors = ['Careca', 'Castanho', 'Loiro', 'Preto', 'Ruivo']
    for hc in hair_colors:
        features[f'Hair_{hc}'] = 1 if hc == hair_color else 0
    
    # Create DataFrame
    df_input = pd.DataFrame([features])
    df_input = df_input[feature_columns]
    
    # Scale
    df_scaled = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0]
    
    return prediction, probability


def main():
    """Main Streamlit application"""
    st.title("🏥 Diabetes Prediction Dashboard")
    st.markdown("---")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🔮 Prediction", "📊 Data Overview", "🤖 Model Performance", "📈 Model Comparison"]
    )
    
    # Load data
    try:
        df = load_data()
        df_processed = get_processed_data(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()
    
    # Page routing
    if page == "📊 Data Overview":
        show_data_overview(df, df_processed)
    elif page == "🤖 Model Performance":
        show_model_performance()
    elif page == "🔮 Prediction":
        show_prediction(df_processed)
    elif page == "📈 Model Comparison":
        show_model_comparison()


def show_data_overview(df, df_processed):
    """Display data overview page"""
    st.header("📊 Data Overview")
    
    # Dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Diabetes Cases", df['Diabético'].sum())
    with col4:
        st.metric("No Diabetes", (df['Diabético'] == 0).sum())
    
    st.markdown("---")
    
    # Data preview
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Statistics")
        st.dataframe(df[['Peso', 'Altura', 'Diabético']].describe(), use_container_width=True)
    
    with col2:
        st.subheader("Target Distribution")
        target_counts = df['Diabético'].value_counts()
        fig = px.pie(
            values=target_counts.values,
            names=['No Diabetes', 'Diabetes'],
            title="Diabetes Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations
    st.subheader("Data Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(df, x='Peso', nbins=20, title='Weight Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Altura', nbins=20, title='Height Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # BMI distribution
    if 'BMI' in df_processed.columns:
        fig = px.histogram(df_processed, x='BMI', nbins=20, title='BMI Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    # Hair color distribution
    st.subheader("Hair Color Distribution")
    hair_counts = df['Cor do cabelo'].value_counts()
    fig = px.bar(
        x=hair_counts.index,
        y=hair_counts.values,
        title="Hair Color Distribution",
        labels={'x': 'Hair Color', 'y': 'Count'}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation matrix
    st.subheader("Correlation Matrix")
    numeric_cols = ['Peso', 'Altura', 'Diabético']
    if 'BMI' in df_processed.columns:
        numeric_cols.append('BMI')
    
    corr_matrix = df_processed[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    st.plotly_chart(fig, use_container_width=True)


def show_model_performance():
    """Display model performance page"""
    st.header("🤖 Model Performance")
    
    best_model_info = get_best_model_data()
    
    if best_model_info is None:
        st.warning("No models found. Please run training first.")
        return
    
    st.subheader(f"Best Model: {best_model_info['model_name']}")
    
    # Display metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    metrics = best_model_info['metrics']
    with col1:
        st.metric("Accuracy", f"{metrics.get('test_accuracy', 0):.4f}")
    with col2:
        st.metric("Precision", f"{metrics.get('test_precision', 0):.4f}")
    with col3:
        st.metric("Recall", f"{metrics.get('test_recall', 0):.4f}")
    with col4:
        st.metric("F1-Score", f"{metrics.get('test_f1_score', 0):.4f}")
    with col5:
        st.metric("ROC-AUC", f"{metrics.get('test_roc_auc', 0):.4f}")
    
    # Model parameters
    st.subheader("Model Hyperparameters")
    st.json(best_model_info['params'])


def show_prediction(df_processed):
    """Display prediction page with enhanced user interface"""
    st.header("🔮 Diabetes Risk Prediction")
    st.markdown("Enter your information below to get a personalized diabetes risk assessment.")
    
    best_model_info = get_best_model_data()
    
    if best_model_info is None:
        st.warning("⚠️ No models found. Please run training first using: `python scripts/train.py`")
        return
    
    # Load model and scaler
    try:
        with st.spinner("🔄 Loading model..."):
            model = load_model_from_mlflow(
                best_model_info['run_id'],
                best_model_info['model_name']
            )
        
        # Load scaler
        scaler = ScalerManager()
        scaler_path = project_root / config['models']['scaler_path']
        
        if not scaler_path.exists():
            # Try alternative paths
            alt_paths = [
                project_root / "models" / "scaler.pkl",
                Path.cwd() / "models" / "scaler.pkl",
                project_root / config['models']['scaler_path'],
            ]
            
            scaler_found = False
            for alt_path in alt_paths:
                if alt_path.exists():
                    scaler.load(str(alt_path))
                    scaler_found = True
                    break
            
            if not scaler_found:
                st.error("❌ Scaler not found. Please run training first using: `python scripts/train.py`")
                st.info(f"Looking for scaler at: {scaler_path}")
                return
        else:
            scaler.load(str(scaler_path))
        
        # Get feature columns
        feature_columns = [col for col in df_processed.columns if col != 'Diabético']
        
        st.success("✅ Model and scaler loaded successfully!")
        
    except Exception as e:
        st.error(f"❌ **Error loading model:** {str(e)}")
        st.markdown("---")
        st.info("**Troubleshooting steps:**")
        st.markdown("""
        1. **Run training first:** Execute `python scripts/train.py` to train models and save artifacts
        2. **Check MLflow tracking URI:** Ensure it's set correctly in `configs/config.yaml`
        3. **Verify experiment exists:** Check that the experiment name matches in the config
        4. **Check file paths:** Ensure the scaler file exists at `models/scaler.pkl`
        """)
        st.code("python scripts/train.py", language="bash")
        return
    
    # Create a form for better UX
    with st.form("prediction_form"):
        st.subheader("📝 Enter Your Information")
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Physical Measurements**")
            weight = st.number_input(
                "Weight (kg)", 
                min_value=30.0, 
                max_value=200.0, 
                value=75.0, 
                step=0.1,
                help="Enter your weight in kilograms"
            )
            height = st.number_input(
                "Height (cm)", 
                min_value=100.0, 
                max_value=250.0, 
                value=170.0, 
                step=0.1,
                help="Enter your height in centimeters"
            )
        
        with col2:
            st.markdown("**Personal Information**")
            hair_color = st.selectbox(
                "Hair Color",
                ["Careca", "Castanho", "Loiro", "Preto", "Ruivo"],
                help="Select your hair color"
            )
            
            # Calculate and display BMI
            bmi = weight / ((height / 100) ** 2)
            bmi_category = ""
            if bmi < 18.5:
                bmi_category = "Underweight"
            elif bmi < 25:
                bmi_category = "Normal"
            elif bmi < 30:
                bmi_category = "Overweight"
            else:
                bmi_category = "Obese"
            
            st.metric("BMI", f"{bmi:.2f}", delta=bmi_category)
        
        # Submit button
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True)
    
    # Show prediction results when form is submitted
    if submitted:
        try:
            with st.spinner("🔄 Analyzing your data..."):
                prediction, probability = predict_diabetes(
                    weight, height, hair_color, model, scaler, feature_columns
                )
            
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            # Main result display
            col1, col2, col3 = st.columns(3)
            
            with col1:
                diabetes_prob = probability[1] * 100
                no_diabetes_prob = probability[0] * 100
                
                if prediction == 1:
                    st.error("⚠️ **HIGH RISK**")
                    st.markdown(f"**Risk Level:** High")
                else:
                    st.success("✅ **LOW RISK**")
                    st.markdown(f"**Risk Level:** Low")
            
            with col2:
                st.metric(
                    "Diabetes Probability",
                    f"{diabetes_prob:.2f}%",
                    delta=f"{diabetes_prob - 50:.2f}%" if diabetes_prob > 50 else None
                )
            
            with col3:
                st.metric(
                    "No Diabetes Probability",
                    f"{no_diabetes_prob:.2f}%",
                    delta=f"{no_diabetes_prob - 50:.2f}%" if no_diabetes_prob > 50 else None
                )
            
            # Visualizations
            st.markdown("### 📈 Risk Visualization")
            
            # Create two columns for visualizations
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                # Probability bar chart
                fig_bar = go.Figure(data=[
                    go.Bar(
                        x=['No Diabetes', 'Diabetes'],
                        y=[no_diabetes_prob, diabetes_prob],
                        marker_color=['#2ecc71', '#e74c3c'],
                        text=[f'{no_diabetes_prob:.1f}%', f'{diabetes_prob:.1f}%'],
                        textposition='auto',
                    )
                ])
                fig_bar.update_layout(
                    title="Prediction Probabilities",
                    yaxis_title="Probability (%)",
                    xaxis_title="Outcome",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with viz_col2:
                # Gauge chart for risk level
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = diabetes_prob,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Diabetes Risk (%)"},
                    delta = {'reference': 50},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkred" if diabetes_prob > 50 else "darkgreen"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=400)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Additional information
            st.markdown("### ℹ️ Additional Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.info(f"**Model Used:** {best_model_info['model_name']}")
            
            with info_col2:
                st.info(f"**Model Accuracy:** {best_model_info['metrics'].get('test_accuracy', 0)*100:.1f}%")
            
            with info_col3:
                st.info(f"**BMI Category:** {bmi_category}")
            
            # Risk interpretation
            st.markdown("### 💡 Risk Interpretation")
            if diabetes_prob < 30:
                st.success("**Low Risk:** Your current profile suggests a low risk of diabetes. Continue maintaining a healthy lifestyle with regular exercise and a balanced diet.")
            elif diabetes_prob < 50:
                st.warning("**Moderate Risk:** Your profile indicates a moderate risk. Consider consulting with a healthcare provider and making lifestyle improvements.")
            else:
                st.error("**High Risk:** Your profile suggests a higher risk of diabetes. We strongly recommend consulting with a healthcare provider for a comprehensive assessment and personalized advice.")
            
            # Disclaimer
            st.markdown("---")
            st.caption("⚠️ **Disclaimer:** This prediction is based on machine learning models and should not replace professional medical advice. Please consult with a healthcare provider for accurate diagnosis and treatment.")
            
        except Exception as e:
            st.error(f"❌ Error making prediction: {str(e)}")
            st.exception(e)


def show_model_comparison():
    """Display model comparison page"""
    st.header("📈 Model Comparison")
    
    comparison_df = get_model_comparison_data()
    
    if comparison_df is None:
        st.warning("No model comparison data available. Please run training first.")
        return
    
    # Display comparison table
    st.subheader("Model Performance Comparison")
    st.dataframe(comparison_df, use_container_width=True)
    
    # Visualizations
    st.subheader("Performance Metrics Visualization")
    
    metrics_to_plot = ['Test Accuracy', 'Test Precision', 'Test Recall', 'Test F1-Score', 'Test ROC-AUC']
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=metrics_to_plot,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}, None]]
    )
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, metric in enumerate(metrics_to_plot):
        row = idx // 3 + 1
        col = idx % 3 + 1
        
        fig.add_trace(
            go.Bar(
                x=comparison_df['Model'],
                y=comparison_df[metric],
                name=metric,
                marker_color=colors[idx % len(colors)]
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Model Performance Metrics"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Heatmap
    st.subheader("Performance Heatmap")
    metrics_matrix = comparison_df.set_index('Model')[metrics_to_plot]
    fig = px.imshow(
        metrics_matrix,
        text_auto=True,
        aspect="auto",
        title="Model Performance Heatmap",
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
