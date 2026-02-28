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

# Initialize MLflow - resolve tracking URI relative to project root
tracking_uri = config['mlflow']['tracking_uri']
# If it's a file URI, resolve it relative to project root
if tracking_uri.startswith("file:"):
    # Extract the path part (remove "file:" prefix)
    uri_path = tracking_uri.replace("file:", "").strip()
    if not Path(uri_path).is_absolute():
        # Make it relative to project root
        resolved_uri = (project_root / uri_path).resolve()
        tracking_uri = f"file:{resolved_uri}"
    else:
        tracking_uri = f"file:{Path(uri_path).resolve()}"

mlflow.set_tracking_uri(tracking_uri)
experiment_name = config['mlflow']['experiment_name']

# Debug info (can be removed in production)
st.sidebar.info(f"🔍 MLflow URI: `{tracking_uri}`")
st.sidebar.info(f"🔍 Project Root: `{project_root}`")


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


@st.cache_data
def get_available_models():
    """Get list of available models from local backup and MLflow"""
    available_models = []
    
    # Check local backup directory
    model_dir = project_root / config['models']['model_dir']
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            if model_file.name != "scaler.pkl":
                model_name = model_file.stem.replace('_', '_').replace('-', '_')
                # Convert to proper case
                if 'logistic' in model_name.lower():
                    model_name = "Logistic_Regression"
                elif 'random' in model_name.lower() or 'forest' in model_name.lower():
                    model_name = "Random_Forest"
                elif 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
                    model_name = "XGBoost"
                elif 'svm' in model_name.lower():
                    model_name = "SVM"
                elif 'knn' in model_name.lower():
                    model_name = "KNN"
                
                if model_name not in [m['name'] for m in available_models]:
                    available_models.append({
                        'name': model_name,
                        'source': 'local',
                        'file': str(model_file)
                    })
    
    # Also try to get from MLflow for comparison
    try:
        comparison_df = get_model_comparison_data()
        if comparison_df is not None:
            for model_name in comparison_df['Model'].values:
                if model_name not in [m['name'] for m in available_models]:
                    available_models.append({
                        'name': model_name,
                        'source': 'mlflow'
                    })
    except:
        pass
    
    # Default models if nothing found
    if not available_models:
        available_models = [
            {'name': 'Logistic_Regression', 'source': 'default'},
            {'name': 'Random_Forest', 'source': 'default'},
            {'name': 'XGBoost', 'source': 'default'},
            {'name': 'SVM', 'source': 'default'},
            {'name': 'KNN', 'source': 'default'},
        ]
    
    return available_models


def load_model_from_local_backup(model_name: str):
    """Try to load model from local backup directory"""
    import joblib
    
    # Try different possible model file names and formats
    model_names_to_try = [
        model_name.lower().replace('_', '_'),  # logistic_regression
        model_name.lower(),  # logistic_regression (already lowercase)
        model_name,  # Logistic_Regression
        model_name.replace('_', '-').lower(),  # logistic-regression
    ]
    
    model_dir = project_root / config['models']['model_dir']
    
    for name in model_names_to_try:
        # Try .pkl extension
        model_path = model_dir / f"{name}.pkl"
        if model_path.exists():
            try:
                model = joblib.load(str(model_path))
                st.sidebar.success(f"✅ Loaded {model_name} from local backup")
                return model
            except Exception as e:
                st.sidebar.warning(f"⚠️ Could not load {name}.pkl: {str(e)}")
                continue
        
        # Try .joblib extension
        model_path = model_dir / f"{name}.joblib"
        if model_path.exists():
            try:
                model = joblib.load(str(model_path))
                st.sidebar.success(f"✅ Loaded {model_name} from local backup")
                return model
            except Exception as e:
                continue
    
    return None


def load_model_from_mlflow(run_id: str, model_name: str):
    """Load model from MLflow with error handling and fallback"""
    # First try local backup
    local_model = load_model_from_local_backup(model_name)
    if local_model is not None:
        return local_model
    
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
    
    try:
        # First try the runs:/ URI format
        model_uri = f"runs:/{run_id}/model"
        
        if 'XGBoost' in model_name:
            return mlflow.xgboost.load_model(model_uri)
        else:
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        # If runs:/ format fails, try using direct artifact path
        try:
            # Get the run to find artifact path
            run = client.get_run(run_id)
            artifact_path = run.info.artifact_uri
            
            # Try direct file path
            if artifact_path.startswith("file://"):
                artifact_path = artifact_path.replace("file://", "")
            
            model_path = Path(artifact_path) / "model"
            
            # Check if model directory exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found at: {model_path}")
            
            # Try loading from direct path
            if 'XGBoost' in model_name:
                return mlflow.xgboost.load_model(str(model_path))
            else:
                return mlflow.sklearn.load_model(str(model_path))
                
        except Exception as direct_error:
            # If direct path also fails, try to get the latest run
            st.warning(f"⚠️ Could not load model from run {run_id}. Trying to find latest model...")
            
            try:
                experiment = client.get_experiment_by_name(experiment_name)
                
                if experiment is None:
                    # Try to list all experiments for debugging
                    try:
                        experiments = client.search_experiments()
                        exp_names = [exp.name for exp in experiments]
                        raise ValueError(
                            f"Experiment '{experiment_name}' not found. "
                            f"Available experiments: {exp_names}"
                        )
                    except Exception as list_error:
                        raise ValueError(
                            f"Experiment '{experiment_name}' not found. "
                            f"Error listing experiments: {str(list_error)}"
                        )
                
                # Get all runs sorted by creation time (newest first)
                runs = client.search_runs(
                    experiment.experiment_id,
                    order_by=["start_time DESC"],
                    max_results=10  # Get more runs to try
                )
                
                if not runs:
                    raise ValueError(f"No runs found in experiment '{experiment_name}'")
                
                # Get the tracking URI base path
                tracking_uri_base = mlflow.get_tracking_uri()
                if tracking_uri_base.startswith("file:"):
                    tracking_uri_base = tracking_uri_base.replace("file:", "").strip()
                    if not Path(tracking_uri_base).is_absolute():
                        tracking_uri_base = (project_root / tracking_uri_base).resolve()
                    else:
                        tracking_uri_base = Path(tracking_uri_base).resolve()
                else:
                    tracking_uri_base = project_root / "mlruns"
                
                # Try each run until one works
                last_error = None
                for run in runs:
                    try:
                        run_id_to_try = run.info.run_id
                        run_model_name = run.data.tags.get('mlflow.runName', run_id_to_try)
                        
                        st.info(f"🔄 Trying model: {run_model_name} (Run ID: {run_id_to_try})")
                        
                        # Try runs:/ URI first
                        try:
                            model_uri = f"runs:/{run_id_to_try}/model"
                            if 'XGBoost' in run_model_name:
                                model = mlflow.xgboost.load_model(model_uri)
                            else:
                                model = mlflow.sklearn.load_model(model_uri)
                            st.success(f"✅ Successfully loaded model: {run_model_name}")
                            return model
                        except:
                            # If runs:/ fails, try constructing path from tracking URI
                            # Format: mlruns/{experiment_id}/{run_id}/artifacts/model
                            experiment_id = experiment.experiment_id
                            
                            # Try path relative to tracking URI base
                            model_path = tracking_uri_base / str(experiment_id) / run_id_to_try / "artifacts" / "model"
                            
                            if not model_path.exists():
                                # Try alternative: use artifact_uri but resolve relative to project
                                artifact_uri = run.info.artifact_uri
                                if artifact_uri.startswith("file://"):
                                    artifact_uri = artifact_uri.replace("file://", "")
                                
                                artifact_path = Path(artifact_uri)
                                
                                # If artifact path is absolute but doesn't exist, try relative to project root
                                if artifact_path.is_absolute() and not artifact_path.exists():
                                    # Extract just the run directory name
                                    run_dir_name = run_id_to_try
                                    model_path = tracking_uri_base / str(experiment_id) / run_dir_name / "artifacts" / "model"
                                else:
                                    model_path = artifact_path / "model"
                            
                            if model_path.exists():
                                if 'XGBoost' in run_model_name:
                                    model = mlflow.xgboost.load_model(str(model_path))
                                else:
                                    model = mlflow.sklearn.load_model(str(model_path))
                                st.success(f"✅ Successfully loaded model: {run_model_name} (from direct path)")
                                return model
                            else:
                                raise FileNotFoundError(f"Model not found at {model_path}")
                        
                    except Exception as run_error:
                        last_error = run_error
                        continue
                
                # If all runs failed, check if mlruns directory exists and has content
                mlruns_path = tracking_uri_base
                has_mlruns = mlruns_path.exists() if isinstance(mlruns_path, Path) else False
                
                error_details = f"Tried {len(runs)} runs but none could be loaded.\n"
                error_details += f"Last error: {str(last_error)}\n\n"
                
                if not has_mlruns:
                    error_details += f"❌ MLruns directory not found at: {mlruns_path}\n"
                else:
                    error_details += f"✅ MLruns directory exists at: {mlruns_path}\n"
                    # Check if any artifacts exist
                    artifact_dirs = list(mlruns_path.glob("*/artifacts/model"))
                    if artifact_dirs:
                        error_details += f"⚠️ Found {len(artifact_dirs)} model artifact directories, but couldn't load them.\n"
                    else:
                        error_details += f"❌ No model artifact directories found.\n"
                
                error_details += "\n**Solution:** Please run training to create model artifacts:\n"
                error_details += "```bash\npython scripts/train.py\n```"
                
                raise Exception(error_details)
                    
            except Exception as fallback_error:
                # Provide detailed error information
                error_msg = "❌ **Model Loading Failed**\n\n"
                error_msg += f"**Original run ID:** {run_id}\n"
                error_msg += f"**Direct path error:** {str(direct_error)}\n"
                error_msg += f"**Fallback error:** {str(fallback_error)}\n\n"
                error_msg += "**Environment Info:**\n"
                error_msg += f"- MLflow Tracking URI: `{mlflow.get_tracking_uri()}`\n"
                error_msg += f"- Experiment Name: `{experiment_name}`\n"
                error_msg += f"- Project Root: `{project_root}`\n\n"
                error_msg += "**🔧 Solution:**\n\n"
                error_msg += "The model artifacts are missing. Please retrain the models:\n\n"
                error_msg += "```bash\n"
                error_msg += "# Make sure you're in the project root directory\n"
                error_msg += "cd \"C:\\Users\\EduardoSulz\\OneDrive - OHI\\Documents\\Projects\\AAAA Projeto Pessoal\\projeto-diabetes\"\n\n"
                error_msg += "# Activate your conda environment\n"
                error_msg += "conda activate base\n\n"
                error_msg += "# Run training\n"
                error_msg += "python scripts/train.py\n"
                error_msg += "```\n\n"
                error_msg += "This will:\n"
                error_msg += "1. Train all models\n"
                error_msg += "2. Save model artifacts to MLflow\n"
                error_msg += "3. Save the scaler for predictions\n\n"
                error_msg += "After training completes, refresh this page."
                
                raise Exception(error_msg)


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


def get_available_models_list():
    """Get list of available model names"""
    available_models = []
    
    # Check local backup directory
    model_dir = project_root / config['models']['model_dir']
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            if model_file.name != "scaler.pkl":
                model_name = model_file.stem
                # Normalize model name
                if 'logistic' in model_name.lower():
                    model_name = "Logistic_Regression"
                elif 'random' in model_name.lower() or 'forest' in model_name.lower():
                    model_name = "Random_Forest"
                elif 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
                    model_name = "XGBoost"
                elif 'svm' in model_name.lower():
                    model_name = "SVM"
                elif 'knn' in model_name.lower():
                    model_name = "KNN"
                
                if model_name not in available_models:
                    available_models.append(model_name)
    
    # Also try to get from MLflow
    try:
        comparison_df = get_model_comparison_data()
        if comparison_df is not None:
            for model_name in comparison_df['Model'].values:
                if model_name not in available_models:
                    available_models.append(model_name)
    except:
        pass
    
    # Default models if nothing found
    if not available_models:
        available_models = ['Logistic_Regression', 'Random_Forest', 'XGBoost', 'SVM', 'KNN']
    
    return sorted(available_models)


def show_prediction(df_processed):
    """Display prediction page with enhanced user interface"""
    st.header("🔮 Diabetes Risk Prediction")
    st.markdown("Enter your information below to get a personalized diabetes risk assessment.")
    
    # Get available models and best model info
    available_models = get_available_models_list()
    best_model_info = get_best_model_data()
    
    if not available_models:
        st.warning("⚠️ No models found. Please run training first using: `python scripts/train.py`")
        return
    
    # Model selection dropdown
    st.subheader("🤖 Model Selection")
    
    # Set default to best model if available, otherwise first in list
    default_model = best_model_info['model_name'] if best_model_info and best_model_info['model_name'] in available_models else available_models[0]
    default_index = available_models.index(default_model) if default_model in available_models else 0
    
    selected_model_name = st.selectbox(
        "Choose a model for prediction:",
        options=available_models,
        index=default_index,
        help="Select which machine learning model to use for the diabetes risk prediction. The best model is selected by default."
    )
    
    # Show model info
    col1, col2 = st.columns(2)
    with col1:
        if selected_model_name == default_model and best_model_info:
            st.info(f"⭐ **Best Model** (ROC-AUC: {best_model_info['metrics'].get('test_roc_auc', 0):.4f})")
        else:
            # Try to get metrics for selected model
            try:
                comparison_df = get_model_comparison_data()
                if comparison_df is not None:
                    model_metrics = comparison_df[comparison_df['Model'] == selected_model_name]
                    if not model_metrics.empty:
                        roc_auc = model_metrics.iloc[0]['Test ROC-AUC']
                        st.info(f"📊 ROC-AUC: {roc_auc:.4f}")
            except:
                pass
    
    with col2:
        # Check if model exists locally
        model_dir = project_root / config['models']['model_dir']
        model_file = model_dir / f"{selected_model_name.lower().replace('_', '_')}.pkl"
        if model_file.exists():
            st.success("💾 Available locally")
        else:
            st.info("☁️ Loading from MLflow")
    
    # Load selected model and scaler
    try:
        with st.spinner(f"🔄 Loading {selected_model_name}..."):
            # Try local backup first (works better for Streamlit Cloud)
            model = load_model_from_local_backup(selected_model_name)
            
            # If local backup doesn't exist, try MLflow
            if model is None:
                st.info("📦 Local backup not found, trying MLflow...")
                # Try to get run ID for this model from MLflow
                try:
                    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())
                    experiment = client.get_experiment_by_name(experiment_name)
                    if experiment:
                        runs = client.search_runs(
                            experiment.experiment_id,
                            filter_string=f"tags.mlflow.runName = '{selected_model_name}'",
                            max_results=1
                        )
                        if runs:
                            run_id = runs[0].info.run_id
                            model = load_model_from_mlflow(run_id, selected_model_name)
                        else:
                            # Fallback: try best model if selected not found
                            if best_model_info:
                                st.warning(f"⚠️ {selected_model_name} not found in MLflow. Using best model instead.")
                                model = load_model_from_mlflow(best_model_info['run_id'], best_model_info['model_name'])
                            else:
                                raise Exception(f"Model {selected_model_name} not found in MLflow and no best model available.")
                except Exception as mlflow_error:
                    st.error(f"❌ Could not load model: {str(mlflow_error)}")
                    st.warning("💡 Please ensure models are trained and saved locally.")
                    return
        
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
