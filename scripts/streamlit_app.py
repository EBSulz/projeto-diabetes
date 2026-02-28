"""Streamlit application for diabetes prediction visualization and interaction"""
import sys
from pathlib import Path

# Add project root to path (go up one level from scripts/ to project root)
project_root = Path(__file__).parent.parent
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
from src.utils.config import load_config, get_project_root

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
project_root = get_project_root()
config = load_config(project_root / "configs" / "config.yaml")

# Initialize MLflow
mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
experiment_name = config['mlflow']['experiment_name']


@st.cache_data
def load_data():
    """Load and cache the dataset"""
    data_path = project_root / config['data']['raw_data_path']
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
    """Load model from MLflow"""
    model_uri = f"runs:/{run_id}/model"
    
    if 'XGBoost' in model_name:
        return mlflow.xgboost.load_model(model_uri)
    else:
        return mlflow.sklearn.load_model(model_uri)


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
        ["📊 Data Overview", "🤖 Model Performance", "🔮 Prediction", "📈 Model Comparison"]
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
    """Display prediction page"""
    st.header("🔮 Diabetes Prediction")
    
    best_model_info = get_best_model_data()
    
    if best_model_info is None:
        st.warning("No models found. Please run training first.")
        return
    
    # Load model and scaler
    try:
        model = load_model_from_mlflow(
            best_model_info['run_id'],
            best_model_info['model_name']
        )
        
        scaler = ScalerManager()
        scaler_path = project_root / config['models']['scaler_path']
        if scaler_path.exists():
            scaler.load(str(scaler_path))
        else:
            st.error("Scaler not found. Please run training first.")
            return
        
        # Get feature columns
        feature_columns = [col for col in df_processed.columns if col != 'Diabético']
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Input form
    st.subheader("Enter Patient Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=75.0, step=0.1)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
    
    with col2:
        hair_color = st.selectbox(
            "Hair Color",
            ["Careca", "Castanho", "Loiro", "Preto", "Ruivo"]
        )
        bmi = weight / ((height / 100) ** 2)
        st.metric("BMI", f"{bmi:.2f}")
    
    # Predict button
    if st.button("Predict Diabetes Risk", type="primary"):
        try:
            prediction, probability = predict_diabetes(
                weight, height, hair_color, model, scaler, feature_columns
            )
            
            st.markdown("---")
            st.subheader("Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **High Risk of Diabetes**")
                else:
                    st.success("✅ **Low Risk of Diabetes**")
            
            with col2:
                st.metric(
                    "Probability of Diabetes",
                    f"{probability[1] * 100:.2f}%"
                )
            
            # Probability visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['No Diabetes', 'Diabetes'],
                    y=[probability[0] * 100, probability[1] * 100],
                    marker_color=['green', 'red']
                )
            ])
            fig.update_layout(
                title="Prediction Probabilities",
                yaxis_title="Probability (%)",
                xaxis_title="Outcome"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


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
